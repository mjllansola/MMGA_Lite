"""Manual chirp (dispersion) correction dialog.

The user marks the time-zero curve **by hand**: left-click the carpet to add a
(wavelength, t0) control point, right-click to remove the nearest one. A
polynomial is fitted through the points and each wavelength column is shifted
so all columns share a common time zero. There is deliberately no auto-guess.
"""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox,
    QSpinBox, QDoubleSpinBox, QSizePolicy, QFrame, QDialogButtonBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QWidget,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from core.data import TADataset
from core.dispersion import fit_chirp_poly, apply_dispersion_correction


def _symlog_time(ax, axis: str, break_ps: float, on: bool) -> None:
    """Apply a symlog (linear near 0, log beyond ``break_ps``) or linear scale
    to one axis, so the region around t0 is easy to see and click."""
    scale = "symlog" if on else "linear"
    if axis == "x":
        ax.set_xscale(scale, **({"linthresh": max(break_ps, 1e-6)} if on else {}))
    else:
        ax.set_yscale(scale, **({"linthresh": max(break_ps, 1e-6)} if on else {}))


class _PreciseTraceDialog(QDialog):
    """Pop-up kinetic trace for placing one t0 precisely by clicking on it."""

    def __init__(self, time, kinetic, wl, t_guess, x_label="λ", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select precise time zero")
        self.resize(620, 500)
        self._time = np.asarray(time, dtype=float)
        self._kinetic = np.asarray(kinetic, dtype=float)
        self._wl = wl
        self._t_sel = float(t_guess)
        self._vline = None

        lay = QVBoxLayout(self)
        self._fig = Figure(figsize=(5.5, 3.6), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._canvas, 1)
        self._canvas.mpl_connect("button_press_event", self._on_click)

        row = QHBoxLayout()
        self._chk_semi = QCheckBox("Semilog time")
        self._chk_semi.setChecked(True)
        self._chk_semi.stateChanged.connect(self._draw)
        row.addWidget(self._chk_semi)
        row.addWidget(QLabel("Break (ps):"))
        self._spn_break = QDoubleSpinBox()
        self._spn_break.setRange(0.001, 1e4)
        self._spn_break.setValue(1.0)
        self._spn_break.setDecimals(3)
        self._spn_break.valueChanged.connect(self._draw)
        row.addWidget(self._spn_break)
        row.addSpacing(12)
        row.addWidget(QLabel("t0:"))
        self._lbl = QLabel(f"{self._t_sel:.5g}")
        self._lbl.setStyleSheet("font-weight: bold;")
        row.addWidget(self._lbl)
        row.addStretch(1)
        lay.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self._draw()

    def _draw(self, *_):
        self._ax.clear()
        self._ax.plot(self._time, self._kinetic, "s-", color="C3", ms=3.5,
                      lw=1.0)
        self._ax.set_xlabel("Time (ps)")
        self._ax.set_ylabel("ΔOD")
        self._ax.set_title(f"{self._wl:.5g} — click to place t0", fontsize=9)
        self._ax.grid(True, alpha=0.3)
        self._vline = self._ax.axvline(self._t_sel, color="#444", lw=1.8,
                                       ls="--")
        _symlog_time(self._ax, "x", self._spn_break.value(),
                     self._chk_semi.isChecked())
        self._canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self._ax or event.xdata is None:
            return
        idx = int(np.argmin(np.abs(self._time - event.xdata)))
        self._t_sel = float(self._time[idx])
        self._lbl.setText(f"{self._t_sel:.5g}")
        if self._vline is not None:
            self._vline.set_xdata([self._t_sel, self._t_sel])
        self._canvas.draw_idle()

    def t_selected(self) -> float:
        return self._t_sel


class ChirpDialog(QDialog):
    """Manual chirp-correction dialog for one TA matrix.

    On accept, :meth:`corrected_dataset` returns a new, chirp-corrected
    :class:`TADataset` (or ``None`` if no valid curve was defined).
    """

    def __init__(self, ds: TADataset, x_label="Wavelength (nm)", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Remove chirp — {ds.name}")
        self.resize(1040, 720)
        self._ds = ds
        self._x_label = x_label
        self._points: list[tuple[float, float]] = []
        self._coeffs = None
        self._scatter = None
        self._poly_line = None
        self._corrected = None
        self._build()
        self._draw_carpet()

    # ------------------------------------------------------------------

    def _build(self):
        root = QVBoxLayout(self)

        info = QLabel(
            "Left-click the carpet to add a time-zero point, right-click to "
            "remove the nearest. A polynomial is fitted through your points; "
            "click Apply to shift every wavelength to a common t0. "
            "Points are placed manually — there is no auto-detection.")
        info.setWordWrap(True)
        root.addWidget(info)

        main = QHBoxLayout()

        # ---- carpet ----------------------------------------------------
        self._fig = Figure(figsize=(7.5, 5.0), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        main.addWidget(self._canvas, 3)

        # ---- control panel --------------------------------------------
        side = QVBoxLayout()

        self._chk_precise = QCheckBox("Fine picking (show trace)")
        self._chk_precise.setChecked(True)
        self._chk_precise.setToolTip(
            "When on, a left-click opens the kinetic trace at that wavelength "
            "so you can click the exact t0.")
        side.addWidget(self._chk_precise)

        semi = QHBoxLayout()
        self._chk_semi = QCheckBox("Semilog time")
        self._chk_semi.setChecked(True)
        self._chk_semi.stateChanged.connect(self._draw_carpet)
        semi.addWidget(self._chk_semi)
        semi.addWidget(QLabel("Break:"))
        self._spn_break = QDoubleSpinBox()
        self._spn_break.setRange(0.001, 1e4)
        self._spn_break.setValue(1.0)
        self._spn_break.setDecimals(3)
        self._spn_break.valueChanged.connect(self._draw_carpet)
        semi.addWidget(self._spn_break)
        side.addLayout(semi)

        order = QHBoxLayout()
        order.addWidget(QLabel("Polynomial order:"))
        self._spn_order = QSpinBox()
        self._spn_order.setRange(1, 8)
        self._spn_order.setValue(3)
        self._spn_order.valueChanged.connect(self._refit)
        order.addWidget(self._spn_order)
        order.addStretch(1)
        side.addLayout(order)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels([self._x_label, "t0 (ps)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        side.addWidget(self._table, 1)

        row = QHBoxLayout()
        b_rm = QPushButton("Remove point")
        b_rm.clicked.connect(self._remove_selected)
        b_undo = QPushButton("Undo last")
        b_undo.clicked.connect(self._undo_last)
        b_clr = QPushButton("Clear all")
        b_clr.clicked.connect(self._clear)
        row.addWidget(b_rm)
        row.addWidget(b_undo)
        row.addWidget(b_clr)
        side.addLayout(row)

        self._lbl_coeffs = QLabel("Add ≥ 2 points to fit a curve.")
        self._lbl_coeffs.setWordWrap(True)
        self._lbl_coeffs.setStyleSheet("color: #555; font-size: 9pt;")
        side.addWidget(self._lbl_coeffs)

        panel = QWidget()
        panel.setLayout(side)
        panel.setMaximumWidth(320)
        main.addWidget(panel, 0)
        root.addLayout(main, 1)

        # ---- apply / cancel -------------------------------------------
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        root.addWidget(sep)
        self._btns = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._apply_btn = self._btns.addButton("Apply correction",
                                               QDialogButtonBox.AcceptRole)
        self._apply_btn.setEnabled(False)
        self._btns.accepted.connect(self._on_apply)
        self._btns.rejected.connect(self.reject)
        root.addWidget(self._btns)

    # ------------------------------------------------------------------

    def _draw_carpet(self, *_):
        ds = self._ds
        self._ax.clear()
        self._scatter = None
        self._poly_line = None
        vmax = float(np.nanmax(np.abs(ds.data))) or 1.0
        self._ax.pcolormesh(ds.wavelength, ds.time, np.nan_to_num(ds.data),
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            shading="auto")
        self._ax.set_xlabel(self._x_label)
        self._ax.set_ylabel("Time (ps)")
        _symlog_time(self._ax, "y", self._spn_break.value(),
                     self._chk_semi.isChecked())
        self._refit()

    def _refit(self, *_):
        # scatter
        if self._scatter is not None:
            try:
                self._scatter.remove()
            except (ValueError, AttributeError):
                pass
            self._scatter = None
        if self._points:
            self._scatter = self._ax.scatter(
                [p[0] for p in self._points], [p[1] for p in self._points],
                s=60, marker="s", c="#ff8c00", edgecolors="#cc4400",
                linewidths=1.0, zorder=10)

        # polynomial curve
        if self._poly_line is not None:
            try:
                self._poly_line.remove()
            except (ValueError, AttributeError):
                pass
            self._poly_line = None

        self._coeffs = None
        if len(self._points) >= 2:
            try:
                self._coeffs = fit_chirp_poly(self._points,
                                              self._spn_order.value())
                wl_fine = np.linspace(float(self._ds.wavelength.min()),
                                      float(self._ds.wavelength.max()), 600)
                self._poly_line, = self._ax.plot(
                    wl_fine, np.polyval(self._coeffs, wl_fine), "-",
                    color="k", lw=1.8, zorder=5)
                self._lbl_coeffs.setText(
                    "t0(λ) = " + " , ".join(f"{c:.4g}" for c in self._coeffs))
            except Exception as exc:
                self._lbl_coeffs.setText(f"Fit error: {exc}")
        else:
            self._lbl_coeffs.setText("Add ≥ 2 points to fit a curve.")

        self._apply_btn.setEnabled(self._coeffs is not None)
        self._rebuild_table()
        self._canvas.draw_idle()

    def _rebuild_table(self):
        self._table.setRowCount(0)
        for wl, t0 in self._points:
            r = self._table.rowCount()
            self._table.insertRow(r)
            self._table.setItem(r, 0, QTableWidgetItem(f"{wl:.5g}"))
            self._table.setItem(r, 1, QTableWidgetItem(f"{t0:.5g}"))

    # ------------------------------------------------------------------

    def _on_click(self, event):
        if event.inaxes != self._ax or event.xdata is None or event.ydata is None:
            return
        ds = self._ds
        wl_idx = int(np.argmin(np.abs(ds.wavelength - event.xdata)))
        wl = float(ds.wavelength[wl_idx])

        if event.button == 1:                       # left: add
            t_idx = int(np.argmin(np.abs(ds.time - event.ydata)))
            t0 = float(ds.time[t_idx])
            if self._chk_precise.isChecked():
                dlg = _PreciseTraceDialog(ds.time, ds.data[:, wl_idx], wl, t0,
                                          x_label=self._x_label, parent=self)
                dlg._chk_semi.setChecked(self._chk_semi.isChecked())
                dlg._spn_break.setValue(self._spn_break.value())
                dlg._draw()
                if dlg.exec() != QDialog.Accepted:
                    return
                t0 = dlg.t_selected()
            # Replace any existing point at (near) this wavelength.
            span = float(ds.wavelength.max() - ds.wavelength.min())
            tol = max(span / max(ds.nw, 1) * 2.0, 1e-9)
            self._points = [(w, v) for (w, v) in self._points
                            if abs(w - wl) > tol]
            self._points.append((wl, t0))
            self._points.sort(key=lambda p: p[0])
            self._refit()
        elif event.button == 3:                     # right: remove nearest
            if not self._points:
                return
            span = float(ds.wavelength.max() - ds.wavelength.min()) or 1.0
            dists = [abs(w - event.xdata) for (w, _) in self._points]
            if min(dists) < span * 0.08:
                self._points.pop(int(np.argmin(dists)))
                self._refit()

    def _remove_selected(self):
        rows = sorted({i.row() for i in self._table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            if 0 <= r < len(self._points):
                self._points.pop(r)
        if rows:
            self._refit()

    def _undo_last(self):
        if self._points:
            self._points.pop()
            self._refit()

    def _clear(self):
        if self._points:
            self._points.clear()
            self._refit()

    # ------------------------------------------------------------------

    def _on_apply(self):
        if self._coeffs is None:
            return
        ds = self._ds
        data = apply_dispersion_correction(ds.data, ds.time, ds.wavelength,
                                           self._coeffs)
        out = TADataset.from_numpy(ds.time, ds.wavelength, data, ds.name)
        out.x_unit = ds.x_unit
        self._corrected = out
        self.accept()

    def corrected_dataset(self):
        """Return the chirp-corrected dataset, or None if none was applied."""
        return self._corrected
