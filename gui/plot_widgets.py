"""Reusable matplotlib widgets: an exportable figure and a matrix viewer.

Every plot in the application is wrapped in :class:`ExportableFigure`, which
provides a matplotlib navigation toolbar (PNG / SVG / PDF export, zoom, pan)
and an optional "Export data (CSV)" button so both the picture and the
underlying numbers can be saved.

Kinetics are drawn on a split time axis -- linear from ``KIN_T_START`` to
``KIN_T_SPLIT`` and logarithmic beyond ``KIN_T_SPLIT`` -- the usual convention
for ultrafast transient-absorption traces.
"""

import csv

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QFileDialog, QSizePolicy,
)

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT,
)


# ---------------------------------------------------------------------------
# Kinetics time-axis convention
# ---------------------------------------------------------------------------
# The time axis for every kinetics view is a broken axis: a linear panel from
# KIN_T_START to KIN_T_SPLIT (so the rise around t0 is resolved) followed by a
# logarithmic panel from KIN_T_SPLIT to the end of the trace.
KIN_T_START = -3.0     # ps -- left edge of the linear panel
KIN_T_SPLIT = 1.0      # ps -- linear below this, logarithmic above
_LINLOG_WR = (1.0, 2.0)   # width ratio of the linear : log panels
# Centre of the panel pair, expressed as a fraction of the linear panel's
# width (used to place a single centred x-label / title over both panels).
_LINLOG_CENTER = (sum(_LINLOG_WR) / 2.0) / _LINLOG_WR[0]


def make_linlog_axes(fig, subplotspec=None):
    """Create a (linear, log) pair of time axes sharing a y-axis.

    If *subplotspec* is given the pair is nested inside that cell; otherwise it
    fills the whole figure. Returns ``(ax_lin, ax_log)``.
    """
    if subplotspec is None:
        gs = fig.add_gridspec(1, 2, width_ratios=_LINLOG_WR, wspace=0.045)
        ax_lin = fig.add_subplot(gs[0])
        ax_log = fig.add_subplot(gs[1], sharey=ax_lin)
    else:
        inner = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=subplotspec,
            width_ratios=_LINLOG_WR, wspace=0.045)
        ax_lin = fig.add_subplot(inner[0])
        ax_log = fig.add_subplot(inner[1], sharey=ax_lin)
    return ax_lin, ax_log


def draw_linlog(ax_lin, ax_log, t, y, **kw):
    """Plot one series (t, y) across the linear and log panels.

    The label (for a legend) is kept only on the log panel so it appears once.
    Non-positive times are dropped on the log panel to avoid log-scale warnings.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    kw_lin = dict(kw)
    kw_lin.pop("label", None)
    ax_lin.plot(t, y, **kw_lin)
    m = t > 0
    ax_log.plot(t[m], y[m], **kw)


def style_linlog(ax_lin, ax_log, t_max, ylabel=None, title=None, legend=False):
    """Apply the broken-axis cosmetics after the series have been drawn."""
    ax_lin.set_xlim(KIN_T_START, KIN_T_SPLIT)
    ax_log.set_xscale("log")
    ax_log.set_xlim(KIN_T_SPLIT, max(float(t_max), KIN_T_SPLIT * 10.0))

    # Hide the facing spines / duplicate y ticks so the pair reads as one axis.
    ax_lin.spines["right"].set_visible(False)
    ax_log.spines["left"].set_visible(False)
    ax_log.tick_params(axis="y", which="both", left=False, labelleft=False)
    for ax in (ax_lin, ax_log):
        ax.axhline(0, color="gray", lw=0.6, zorder=0)

    if ylabel:
        ax_lin.set_ylabel(ylabel)
    # A single x-label centred under both panels.
    ax_lin.set_xlabel("Time (ps)")
    ax_lin.xaxis.set_label_coords(_LINLOG_CENTER, -0.16)
    if title:
        ax_lin.text(_LINLOG_CENTER, 1.02, title, transform=ax_lin.transAxes,
                    ha="center", va="bottom", fontsize=9)
    if legend:
        ax_log.legend(fontsize=8, ncol=2)


def style_carpet_time(ax, t_max, axis="y"):
    """Style a 2-D carpet's time axis: symlog (linear near 0, log beyond
    KIN_T_SPLIT) starting the view at KIN_T_START."""
    if axis == "y":
        ax.set_yscale("symlog", linthresh=KIN_T_SPLIT)
        ax.set_ylim(KIN_T_START, float(t_max))
    else:
        ax.set_xscale("symlog", linthresh=KIN_T_SPLIT)
        ax.set_xlim(KIN_T_START, float(t_max))


class ExportableFigure(QWidget):
    """A matplotlib Figure with a navigation toolbar and an optional CSV export.

    Parameters
    ----------
    figsize : tuple
        Figure size in inches.
    csv_callback : callable | None
        If given, an "Export data (CSV)" button is shown; clicking it prompts
        for a path and calls ``csv_callback(path)``.
    csv_name : str
        Default filename stem offered in the CSV save dialog.
    citation : str | None
        If given, a small right-aligned label shown next to the toolbar
        (e.g. the paper/data reference and preprint link).
    """

    CITATION = ("Bercy et al., J. Am. Chem. Soc. 2026, 148, 23976–23985 "
                "· arXiv: TBA")

    def __init__(self, figsize=(6.0, 4.0), csv_callback=None,
                 csv_name="data", citation=None, parent=None):
        super().__init__(parent)
        self._csv_callback = csv_callback
        self._csv_name = csv_name

        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.toolbar, 1)
        if citation is not None:
            lbl = QLabel(citation)
            lbl.setStyleSheet("color: #ddd; font-size: 13px; font-weight: 600;")
            top.addWidget(lbl, 0)
        if csv_callback is not None:
            btn = QPushButton("Export data (CSV)…")
            btn.clicked.connect(self._on_export_csv)
            top.addWidget(btn, 0)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(top)
        lay.addWidget(self.canvas, 1)

    def _on_export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export data as CSV", f"{self._csv_name}.csv",
            "CSV files (*.csv);;All files (*)")
        if path and self._csv_callback is not None:
            self._csv_callback(path)

    def draw(self):
        self.canvas.draw_idle()


def _write_csv(path: str, header: list, columns: list) -> None:
    """Write a CSV with *header* and *columns* (a list of equal-length 1-D
    arrays, one per column)."""
    ncol = len(columns)
    nrow = max((len(c) for c in columns), default=0)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in range(nrow):
            row = []
            for c in range(ncol):
                col = columns[c]
                row.append(col[r] if r < len(col) else "")
            w.writerow(row)


def _write_matrix_csv(path: str, time, wavelength, data) -> None:
    """Write a TA matrix as: corner cell blank, wavelengths across the top,
    times down the first column, dOD in the body."""
    time = np.asarray(time)
    wavelength = np.asarray(wavelength)
    data = np.asarray(data)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time\\wavelength"] + [f"{x:g}" for x in wavelength])
        for i in range(data.shape[0]):
            w.writerow([f"{time[i]:g}"] + [f"{v:g}" for v in data[i, :]])


class MatrixViewer(QWidget):
    """Explore one TA matrix: carpet plot plus a spectrum-at-time slice and a
    kinetic-at-wavelength slice, driven by two slide bars.

    This is a pure viewer -- there are no processing controls, only the slide
    bars for moving through the spectra and time traces.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ds = None
        self._t_idx = 0
        self._w_idx = 0
        self._cbar = None

        self.figpanel = ExportableFigure(
            figsize=(8.0, 5.0), csv_callback=self._export_matrix_csv,
            csv_name="matrix", citation=ExportableFigure.CITATION)
        self.fig = self.figpanel.fig
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0])
        self.ax_carpet = self.fig.add_subplot(gs[:, 0])
        self.ax_spec = self.fig.add_subplot(gs[0, 1])
        # Kinetics: split linear/log time axis.
        self.ax_kin_lin, self.ax_kin_log = make_linlog_axes(self.fig, gs[1, 1])

        # ---- slide bars ------------------------------------------------
        self.t_slider = QSlider(Qt.Horizontal)
        self.t_slider.valueChanged.connect(self._on_t)
        self.t_label = QLabel("time: -")
        self.t_label.setMinimumWidth(150)

        self.w_slider = QSlider(Qt.Horizontal)
        self.w_slider.valueChanged.connect(self._on_w)
        self.w_label = QLabel("wavelength: -")
        self.w_label.setMinimumWidth(150)

        trow = QHBoxLayout()
        trow.addWidget(self.t_label)
        trow.addWidget(self.t_slider, 1)
        wrow = QHBoxLayout()
        wrow.addWidget(self.w_label)
        wrow.addWidget(self.w_slider, 1)

        lay = QVBoxLayout(self)
        lay.addWidget(self.figpanel, 1)
        lay.addLayout(trow)
        lay.addLayout(wrow)

    # ------------------------------------------------------------------

    def set_dataset(self, ds):
        self._ds = ds
        if ds is None or ds.nt == 0 or ds.nw == 0:
            for ax in (self.ax_carpet, self.ax_spec,
                       self.ax_kin_lin, self.ax_kin_log):
                ax.clear()
            self.figpanel.draw()
            return
        # Default: time slice near the signal maximum, wavelength at max |dOD|.
        flat = np.nan_to_num(ds.data)
        self._t_idx = int(np.argmax(np.abs(flat).sum(axis=1)))
        self._w_idx = int(np.argmax(np.abs(flat).sum(axis=0)))
        self.t_slider.blockSignals(True)
        self.w_slider.blockSignals(True)
        self.t_slider.setRange(0, ds.nt - 1)
        self.w_slider.setRange(0, ds.nw - 1)
        self.t_slider.setValue(self._t_idx)
        self.w_slider.setValue(self._w_idx)
        self.t_slider.blockSignals(False)
        self.w_slider.blockSignals(False)
        self.redraw()

    def _x_label(self):
        ds = self._ds
        if ds is not None and ds.x_unit:
            return f"Axis ({ds.x_unit})"
        return "Wavelength (nm)"

    def redraw(self):
        ds = self._ds
        if ds is None or ds.nt == 0 or ds.nw == 0:
            return
        t = ds.time
        w = ds.wavelength
        ti = self._t_idx
        wi = self._w_idx
        t_max = float(t.max())

        # ---- carpet ----------------------------------------------------
        ax = self.ax_carpet
        ax.clear()
        vmax = float(np.nanmax(np.abs(ds.data))) or 1.0
        mesh = ax.pcolormesh(w, t, np.nan_to_num(ds.data),
                             cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                             shading="auto")
        ax.axhline(t[ti], color="k", lw=0.8, ls="--")
        ax.axvline(w[wi], color="k", lw=0.8, ls=":")
        ax.set_xlabel(self._x_label())
        ax.set_ylabel("Time (ps)")
        ax.set_title(ds.name, fontsize=9)
        style_carpet_time(ax, t_max, axis="y")
        if self._cbar is None:
            self._cbar = self.fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.02)
        else:
            self._cbar.update_normal(mesh)

        # ---- spectrum at selected time --------------------------------
        ax = self.ax_spec
        ax.clear()
        ax.plot(w, ds.data[ti, :], color="C0")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel(self._x_label())
        ax.set_ylabel("ΔOD")
        ax.set_title(f"Spectrum @ {t[ti]:.3g} ps", fontsize=9)

        # ---- kinetic at selected wavelength (split linear/log time) ---
        self.ax_kin_lin.clear()
        self.ax_kin_log.clear()
        draw_linlog(self.ax_kin_lin, self.ax_kin_log, t, ds.data[:, wi],
                    color="C3")
        style_linlog(self.ax_kin_lin, self.ax_kin_log, t_max, ylabel="ΔOD",
                     title=f"Kinetics @ {w[wi]:.4g}")

        self.t_label.setText(f"time: {t[ti]:.4g} ps  [{ti+1}/{ds.nt}]")
        self.w_label.setText(f"wavelength: {w[wi]:.5g}  [{wi+1}/{ds.nw}]")
        self.figpanel.draw()

    def _on_t(self, v):
        self._t_idx = int(v)
        self.redraw()

    def _on_w(self, v):
        self._w_idx = int(v)
        self.redraw()

    def _export_matrix_csv(self, path):
        ds = self._ds
        if ds is not None:
            _write_matrix_csv(path, ds.time, ds.wavelength, ds.data)
