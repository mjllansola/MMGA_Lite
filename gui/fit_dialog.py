"""Global-fit dialog: set up a kinetic model, fit all loaded matrices with
shared kinetic parameters, and view the results.

Scope is deliberately narrow. There is no multi-matrix spectral linking, no
weighting / constraint editor, no saved-model manager, no species-average
tab and no publication pipeline -- only the essentials of global and target
analysis. Every result view can be exported (figure via the toolbar, numbers
via an "Export data (CSV)" button).
"""

import logging

import numpy as np
import lmfit
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QCheckBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QTabWidget,
    QSlider, QMessageBox, QProgressBar, QAbstractItemView,
)

from core.data import TADataset
from core.models import ParallelModel, SequentialModel, TargetModel
from core.fitting import GlobalFitter, FitResult, transform_spectra
from gui.plot_widgets import (
    ExportableFigure, _write_csv, _write_matrix_csv,
    make_linlog_axes, draw_linlog, style_linlog, style_carpet_time,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker: runs the fit off the UI thread.
# ---------------------------------------------------------------------------

class FitWorker(QObject):
    finished = Signal(object)
    progress = Signal(int, str)
    live_result = Signal(int, object)

    LIVE_EVERY = 5

    def __init__(self, fitter: GlobalFitter, params: lmfit.Parameters):
        super().__init__()
        self._fitter = fitter
        self._params = params

    def run(self):
        def cb(it, params, resid):
            rms = float(np.sqrt(np.mean(resid ** 2)))
            self.progress.emit(it, f"Iter {it}  RMS={rms:.4e}")
            if it % self.LIVE_EVERY == 0:
                try:
                    self.live_result.emit(it, self._fitter.evaluate_at(params))
                except Exception:
                    pass
        try:
            result = self._fitter.run(self._params, progress_cb=cb)
        except Exception as exc:
            logging.getLogger(__name__).exception("Fit engine crashed")
            result = FitResult()
            result.success = False
            result.message = f"internal error - {type(exc).__name__}: {exc}"
        self.finished.emit(result)


# ---------------------------------------------------------------------------

def _pretty_param_label(key: str) -> str:
    """Human-readable label for a parameter key."""
    if key == "t0":
        return "t0 (ps)"
    if key == "irf_fwhm":
        return "IRF FWHM (ps)"
    if key.startswith("log_tau_"):
        rest = key[len("log_tau_"):]
        if rest.endswith("G"):
            return f"τ {rest[:-1]}→gnd (ps)"
        if len(rest) == 2 and rest.isdigit():
            return f"τ {rest[0]}→{rest[1]} (ps)"
        return f"τ{rest} (ps)"
    return key


class FitDialog(QDialog):
    def __init__(self, datasets, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Global fit")
        self.resize(1240, 820)
        self._datasets: list[TADataset] = list(datasets)
        self._model = None
        self._base_params: lmfit.Parameters | None = None
        self._result: FitResult | None = None
        self._thread = None
        self._worker = None
        self._fitter = None

        root = QHBoxLayout(self)
        root.addWidget(self._build_setup_panel(), 0)
        root.addWidget(self._build_results_panel(), 1)

        # App default: a 4-species sequential fit (matches the Raman
        # multi-matrix demo). _on_model_changed then builds the table.
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(1)      # Sequential
        self.model_combo.blockSignals(False)
        self.n_spin.blockSignals(True)
        self.n_spin.setValue(4)
        self.n_spin.blockSignals(False)
        self._on_model_changed()

    # ==================================================================
    # Setup panel
    # ==================================================================

    def _build_setup_panel(self) -> QWidget:
        w = QWidget()
        w.setMaximumWidth(520)
        lay = QVBoxLayout(w)

        # ---- model box -------------------------------------------------
        mbox = QGroupBox("Kinetic model")
        form = QFormLayout(mbox)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Parallel (sum-of-exp)",
            "Sequential (A→B→C…)",
            "Target (compartment)",
        ])
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Model", self.model_combo)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 10)
        self.n_spin.setValue(3)
        self.n_spin.valueChanged.connect(self._on_model_changed)
        form.addRow("Components", self.n_spin)

        self.conn_edit = QLineEdit("1>2, 2>3, 3>4, 4>G")
        self.conn_edit.setToolTip(
            "Target scheme, e.g. '1>2, 2>3, 3>G'. 'a>b' is transfer from "
            "compartment a to b; 'a>G' is decay of a to the ground state.")
        self.conn_edit.editingFinished.connect(self._rebuild_param_table)
        self.conn_row_label = QLabel("Connections")
        form.addRow(self.conn_row_label, self.conn_edit)

        lay.addWidget(mbox)

        # ---- parameter table ------------------------------------------
        pbox = QGroupBox("Parameters (initial guess; lifetimes in ps)")
        pl = QVBoxLayout(pbox)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Parameter", "Value", "Min", "Max", "Vary"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        for c in (1, 2, 3, 4):
            self.table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        pl.addWidget(self.table)
        lay.addWidget(pbox, 1)

        # ---- options + run --------------------------------------------
        self.nonneg_check = QCheckBox("Non-negative spectra (NNLS)")
        lay.addWidget(self.nonneg_check)

        self.run_btn = QPushButton("Run fit")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        rr = QHBoxLayout()
        rr.addWidget(self.run_btn)
        rr.addWidget(self.stop_btn)
        lay.addLayout(rr)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        self.status = QLabel(f"{len(self._datasets)} matrix(es) loaded. "
                             "Kinetics are shared across all of them.")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        return w

    def _on_model_changed(self, *_):
        is_target = self.model_combo.currentIndex() == 2
        self.conn_edit.setVisible(is_target)
        self.conn_row_label.setVisible(is_target)
        self._rebuild_param_table()

    def _make_model(self):
        """Build the model from the UI. Returns (model, n). Raises ValueError
        on an invalid target scheme."""
        idx = self.model_combo.currentIndex()
        n = int(self.n_spin.value())
        if idx == 0:
            m = ParallelModel()
            return m, n
        if idx == 1:
            m = SequentialModel()
            return m, n
        # Target
        conns = self._parse_connections(self.conn_edit.text(), n)
        m = TargetModel()
        m.setup(n, conns)
        return m, n

    @staticmethod
    def _parse_connections(text: str, n: int) -> list:
        """Parse 'a>b, a>G' into a list of (to, from) 0-based pairs.

        'a>b' is transfer a->b, stored as (i=b-1, j=a-1). 'a>G' (ground) is a
        self-connection (a-1, a-1). Indices must lie in 1..n.
        """
        conns = []
        tokens = [t.strip() for t in text.replace("\n", ",").split(",")]
        for tok in tokens:
            if not tok:
                continue
            if ">" not in tok:
                raise ValueError(f"Bad connection '{tok}' (use 'a>b').")
            a_s, b_s = tok.split(">", 1)
            a_s, b_s = a_s.strip(), b_s.strip()
            try:
                a = int(a_s)
            except ValueError:
                raise ValueError(f"Bad source in '{tok}'.")
            if not (1 <= a <= n):
                raise ValueError(f"Source {a} out of range 1..{n} in '{tok}'.")
            if b_s.upper() in ("G", "GND", "GROUND", "0"):
                conns.append((a - 1, a - 1))          # decay to ground
            else:
                try:
                    b = int(b_s)
                except ValueError:
                    raise ValueError(f"Bad target in '{tok}'.")
                if not (1 <= b <= n):
                    raise ValueError(
                        f"Target {b} out of range 1..{n} in '{tok}'.")
                conns.append((b - 1, a - 1))          # transfer a->b
        if not conns:
            raise ValueError("No connections defined.")
        return conns

    # Default starting values and per-lifetime limits (ps) applied when the
    # sequential model is selected -- the app's default fit setup for the
    # lycopene Raman multi-matrix demo. (value, min, max) per species.
    _SEQ_TAU_DEFAULTS = [
        (0.1, 0.1, 1.0),
        (0.5, 0.1, 1.0),
        (5.0, 0.1, 100.0),
        (50.0, 0.1, 1000.0),
    ]
    _SEQ_IRF_DEFAULT = (0.1, 0.1, 0.2)   # (value, min, max) ps

    def _apply_default_bounds(self, model, params) -> None:
        """Seed the sequential model with the app's default lifetimes / limits
        (and IRF), in place. Other models keep their generic model defaults."""
        if not isinstance(model, SequentialModel):
            return
        if "irf_fwhm" in params:
            v, lo, hi = self._SEQ_IRF_DEFAULT
            params["irf_fwhm"].set(value=v, min=lo, max=hi)
        for i, (v, lo, hi) in enumerate(self._SEQ_TAU_DEFAULTS):
            key = f"log_tau_{i+1}"
            if key in params:
                params[key].set(value=float(np.log(v)),
                                min=float(np.log(lo)), max=float(np.log(hi)))

    def _rebuild_param_table(self):
        """Rebuild the model + default parameters and repopulate the table.

        User-entered cells are preserved for parameters that survive the
        rebuild, provided the model type is unchanged — so hand-tuned lifetimes
        are kept when only the component count or the target connections change.
        Switching model type falls back to fresh defaults.
        """
        try:
            model, n = self._make_model()
            base = (model.default_params(n)
                    if not isinstance(model, TargetModel)
                    else model.default_params())
        except ValueError as exc:
            self.status.setText(f"Model error: {exc}")
            return
        self._apply_default_bounds(model, base)

        # Snapshot the current cells (by parameter key) so surviving parameters
        # keep the user's edits. Only within the same model type.
        prior = {}
        if type(model) is type(self._model):
            for r in range(self.table.rowCount()):
                it = self.table.item(r, 0)
                if it is None:
                    continue
                prior[it.data(Qt.UserRole)] = (
                    self.table.item(r, 1).text(), self.table.item(r, 2).text(),
                    self.table.item(r, 3).text(),
                    self.table.item(r, 4).checkState() == Qt.Checked)

        self._model = model
        self._base_params = base

        self.table.setRowCount(0)
        for key, par in base.items():
            if key in ("n_exp", "n_comp"):
                continue
            if key in prior:
                vs, mns, mxs, vary = prior[key]
            elif key.startswith("log_tau_"):
                vs = self._fmt(np.exp(par.value))
                mns = self._fmt(np.exp(par.min))
                mxs = self._fmt(np.exp(par.max))
                vary = bool(par.vary)
            else:
                vs, mns, mxs = (self._fmt(par.value), self._fmt(par.min),
                                self._fmt(par.max))
                vary = bool(par.vary)
            self._add_param_row(key, vs, mns, mxs, vary)
        self.status.setText(
            f"{len(self._datasets)} matrix(es) loaded. "
            "Kinetics are shared across all of them.")

    def _add_param_row(self, key, value_str, min_str, max_str, vary):
        r = self.table.rowCount()
        self.table.insertRow(r)

        name_item = QTableWidgetItem(_pretty_param_label(key))
        name_item.setFlags(Qt.ItemIsEnabled)
        name_item.setData(Qt.UserRole, key)
        self.table.setItem(r, 0, name_item)

        for col, s in ((1, value_str), (2, min_str), (3, max_str)):
            it = QTableWidgetItem(str(s))
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable
                        | Qt.ItemIsSelectable)
            self.table.setItem(r, col, it)

        vary_item = QTableWidgetItem()
        vary_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
                           | Qt.ItemIsSelectable)
        vary_item.setCheckState(Qt.Checked if vary else Qt.Unchecked)
        self.table.setItem(r, 4, vary_item)

    @staticmethod
    def _fmt(v: float) -> str:
        return f"{v:g}"

    def _cell_float(self, r: int, c: int, label: str) -> float:
        txt = self.table.item(r, c).text().strip()
        try:
            return float(txt)
        except ValueError:
            raise ValueError(f"'{txt}' is not a number ({label}).")

    def _build_params(self):
        """Read the table back into an lmfit.Parameters. Returns (model,
        params). Raises ValueError on a bad numeric entry or model."""
        model, n = self._make_model()
        params = (model.default_params(n)
                  if not isinstance(model, TargetModel)
                  else model.default_params())
        for r in range(self.table.rowCount()):
            key = self.table.item(r, 0).data(Qt.UserRole)
            if key not in params:
                continue
            label = _pretty_param_label(key)
            value = self._cell_float(r, 1, label)
            vmin = self._cell_float(r, 2, label)
            vmax = self._cell_float(r, 3, label)
            vary = self.table.item(r, 4).checkState() == Qt.Checked
            is_tau = key.startswith("log_tau_")
            if is_tau and min(value, vmin, vmax) <= 0:
                raise ValueError(f"Lifetime and its limits must be positive "
                                 f"({label}).")
            if vmin > vmax:
                raise ValueError(f"Min must not exceed Max ({label}).")
            if not (vmin <= value <= vmax):
                raise ValueError(
                    f"Value must lie between Min and Max ({label}).")
            if is_tau:
                params[key].set(value=float(np.log(value)),
                                min=float(np.log(vmin)),
                                max=float(np.log(vmax)), vary=vary)
            else:
                params[key].set(value=value, min=vmin, max=vmax, vary=vary)
        return model, params

    # ==================================================================
    # Results panel
    # ==================================================================

    def _build_results_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        top = QHBoxLayout()
        top.addWidget(QLabel("Matrix:"))
        self.matrix_combo = QComboBox()
        self.matrix_combo.addItems([ds.name for ds in self._datasets])
        self.matrix_combo.currentIndexChanged.connect(self._refresh_results)
        top.addWidget(self.matrix_combo, 1)
        lay.addLayout(top)

        self.tabs = QTabWidget()

        # -- Spectra tab --
        spec_tab = QWidget()
        sl = QVBoxLayout(spec_tab)
        vrow = QHBoxLayout()
        vrow.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Native", "SAS", "DAS", "EADS"])
        self.view_combo.currentIndexChanged.connect(self._draw_spectra)
        vrow.addWidget(self.view_combo)
        vrow.addStretch(1)
        sl.addLayout(vrow)
        self.spec_fig = ExportableFigure(csv_callback=self._export_spectra_csv,
                                         csv_name="spectra")
        self.ax_spec = self.spec_fig.fig.add_subplot(111)
        sl.addWidget(self.spec_fig, 1)
        self.tabs.addTab(spec_tab, "Spectra (DAS/SAS/EADS)")

        # -- Fitted traces tab --
        tr_tab = QWidget()
        tl = QVBoxLayout(tr_tab)
        self.trace_fig = ExportableFigure(csv_callback=self._export_trace_csv,
                                          csv_name="fitted_trace")
        self.ax_trace_lin, self.ax_trace_log = make_linlog_axes(
            self.trace_fig.fig)
        tl.addWidget(self.trace_fig, 1)
        wrow = QHBoxLayout()
        self.trace_wl_label = QLabel("wavelength: -")
        self.trace_wl_label.setMinimumWidth(170)
        self.trace_slider = QSlider(Qt.Horizontal)
        self.trace_slider.valueChanged.connect(self._draw_trace)
        wrow.addWidget(self.trace_wl_label)
        wrow.addWidget(self.trace_slider, 1)
        tl.addLayout(wrow)
        self.tabs.addTab(tr_tab, "Fitted traces")

        # -- Concentrations tab --
        conc_tab = QWidget()
        cl = QVBoxLayout(conc_tab)
        self.conc_fig = ExportableFigure(csv_callback=self._export_conc_csv,
                                         csv_name="concentrations")
        self.ax_conc_lin, self.ax_conc_log = make_linlog_axes(
            self.conc_fig.fig)
        cl.addWidget(self.conc_fig, 1)
        self.tabs.addTab(conc_tab, "Concentrations")

        # -- Residuals tab --
        res_tab = QWidget()
        rl = QVBoxLayout(res_tab)
        self.res_fig = ExportableFigure(csv_callback=self._export_resid_csv,
                                        csv_name="residuals")
        self.ax_res = self.res_fig.fig.add_subplot(111)
        rl.addWidget(self.res_fig, 1)
        self.tabs.addTab(res_tab, "Residuals")

        lay.addWidget(self.tabs, 1)
        return w

    def _current_idx(self) -> int:
        return max(0, self.matrix_combo.currentIndex())

    def _x_label(self, ds) -> str:
        if ds.x_unit:
            return f"Axis ({ds.x_unit})"
        return "Wavelength (nm)"

    def _refresh_results(self):
        if self._result is None:
            return
        idx = self._current_idx()
        ds = self._datasets[idx]
        # Reconfigure the trace-tab wavelength slider for this matrix.
        self.trace_slider.blockSignals(True)
        self.trace_slider.setRange(0, max(ds.nw - 1, 0))
        if self.trace_slider.value() >= ds.nw:
            self.trace_slider.setValue(ds.nw // 2)
        self.trace_slider.blockSignals(False)
        self._draw_spectra()
        self._draw_trace()
        self._draw_conc()
        self._draw_resid()

    # ---- spectra ------------------------------------------------------

    def _spectra_view_kind(self) -> str:
        v = self.view_combo.currentText()
        if v == "Native":
            return self._result.display_nomenclature()
        return v

    def _spectra_for_view(self, idx: int):
        r = self._result
        X = r.spectra[idx]
        M = r.eigenmode_M[idx] if idx < len(r.eigenmode_M) else None
        taus = r.eigenmode_taus[idx] if idx < len(r.eigenmode_taus) else None
        kind = self._spectra_view_kind()
        S = transform_spectra(X, kind, r.spectral_type, M, taus, r.labels)
        return kind, S

    def _draw_spectra(self):
        if self._result is None:
            return
        idx = self._current_idx()
        ds = self._datasets[idx]
        kind, S = self._spectra_for_view(idx)
        ax = self.ax_spec
        ax.clear()
        if S is not None:
            r = self._result
            for c in range(S.shape[0]):
                label = r.labels[c] if c < len(r.labels) else f"C{c+1}"
                ax.plot(ds.wavelength, S[c, :], label=label)
            ax.legend(fontsize=8, ncol=2)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel(self._x_label(ds))
        ax.set_ylabel("Amplitude (ΔOD)")
        ax.set_title(f"{kind} — {ds.name}", fontsize=9)
        self.spec_fig.draw()

    # ---- fitted trace -------------------------------------------------

    def _draw_trace(self):
        if self._result is None:
            return
        idx = self._current_idx()
        ds = self._datasets[idx]
        wi = int(self.trace_slider.value())
        wi = min(wi, ds.nw - 1)
        fit = self._result.fitted_data(idx)
        self.ax_trace_lin.clear()
        self.ax_trace_log.clear()
        draw_linlog(self.ax_trace_lin, self.ax_trace_log, ds.time,
                    ds.data[:, wi], marker=".", ms=3, ls="none",
                    color="0.5", label="data")
        if fit is not None:
            draw_linlog(self.ax_trace_lin, self.ax_trace_log, ds.time,
                        fit[:, wi], ls="-", color="C3", lw=1.5, label="fit")
        style_linlog(self.ax_trace_lin, self.ax_trace_log, float(ds.time.max()),
                     ylabel="ΔOD", legend=True,
                     title=f"Trace @ {ds.wavelength[wi]:.5g} — {ds.name}")
        self.trace_wl_label.setText(
            f"wavelength: {ds.wavelength[wi]:.5g}  [{wi+1}/{ds.nw}]")
        self.trace_fig.draw()

    # ---- concentrations ----------------------------------------------

    def _draw_conc(self):
        if self._result is None:
            return
        idx = self._current_idx()
        ds = self._datasets[idx]
        C = self._result.profiles[idx]
        r = self._result
        self.ax_conc_lin.clear()
        self.ax_conc_log.clear()
        for c in range(C.shape[1]):
            label = r.labels[c] if c < len(r.labels) else f"C{c+1}"
            draw_linlog(self.ax_conc_lin, self.ax_conc_log, ds.time,
                        C[:, c], label=label)
        style_linlog(self.ax_conc_lin, self.ax_conc_log, float(ds.time.max()),
                     ylabel="Population", legend=True,
                     title=f"Concentration profiles — {ds.name}")
        self.conc_fig.draw()

    # ---- residuals ----------------------------------------------------

    def _draw_resid(self):
        if self._result is None:
            return
        idx = self._current_idx()
        ds = self._datasets[idx]
        R = self._result.residuals[idx]
        ax = self.ax_res
        ax.clear()
        vmax = float(np.nanmax(np.abs(R))) or 1.0
        mesh = ax.pcolormesh(ds.wavelength, ds.time, np.nan_to_num(R),
                             cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                             shading="auto")
        ax.set_xlabel(self._x_label(ds))
        ax.set_ylabel("Time (ps)")
        style_carpet_time(ax, float(ds.time.max()), axis="y")
        ax.set_title(f"Residuals (data − fit) — {ds.name}", fontsize=9)
        if getattr(self, "_res_cbar", None) is None:
            self._res_cbar = self.res_fig.fig.colorbar(
                mesh, ax=ax, fraction=0.05, pad=0.02)
        else:
            self._res_cbar.update_normal(mesh)
        self.res_fig.draw()

    # ==================================================================
    # CSV exporters
    # ==================================================================

    def _export_spectra_csv(self, path):
        if self._result is None:
            return
        idx = self._current_idx()
        ds = self._datasets[idx]
        kind, S = self._spectra_for_view(idx)
        if S is None:
            return
        labels = self._result.labels
        header = [self._x_label(ds)] + [
            (labels[c] if c < len(labels) else f"C{c+1}")
            for c in range(S.shape[0])]
        cols = [ds.wavelength] + [S[c, :] for c in range(S.shape[0])]
        _write_csv(path, header, cols)

    def _export_trace_csv(self, path):
        if self._result is None:
            return
        idx = self._current_idx()
        ds = self._datasets[idx]
        wi = min(int(self.trace_slider.value()), ds.nw - 1)
        fit = self._result.fitted_data(idx)
        header = ["time_ps", "data", "fit"]
        cols = [ds.time, ds.data[:, wi],
                fit[:, wi] if fit is not None else np.full(ds.nt, np.nan)]
        _write_csv(path, header, cols)

    def _export_conc_csv(self, path):
        if self._result is None:
            return
        idx = self._current_idx()
        ds = self._datasets[idx]
        C = self._result.profiles[idx]
        labels = self._result.labels
        header = ["time_ps"] + [
            (labels[c] if c < len(labels) else f"C{c+1}")
            for c in range(C.shape[1])]
        cols = [ds.time] + [C[:, c] for c in range(C.shape[1])]
        _write_csv(path, header, cols)

    def _export_resid_csv(self, path):
        if self._result is None:
            return
        idx = self._current_idx()
        ds = self._datasets[idx]
        _write_matrix_csv(path, ds.time, ds.wavelength,
                          self._result.residuals[idx])

    # ==================================================================
    # Run / stop
    # ==================================================================

    def _run(self):
        try:
            model, params = self._build_params()
        except ValueError as exc:
            QMessageBox.warning(self, "Fit setup error", str(exc))
            return
        nonneg = self.nonneg_check.isChecked()
        self._fitter = GlobalFitter(self._datasets, model, nonneg=nonneg)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.status.setText("Fitting…")

        self._thread = QThread()
        self._worker = FitWorker(self._fitter, params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # Connect to BOUND METHODS so Qt queues them onto the UI thread.
        self._worker.progress.connect(self._on_progress)
        self._worker.live_result.connect(self._on_live)
        self._worker.finished.connect(self._on_done)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _stop(self):
        if self._fitter is not None:
            self._fitter.abort()
        self.stop_btn.setEnabled(False)
        self.status.setText("Stopping…")

    def _on_progress(self, it, msg):
        self.status.setText(msg)

    def _on_live(self, it, preview):
        # Light live preview: only refresh once we have somewhere to draw.
        if preview is not None and preview.spectra:
            self._result = preview
            self._draw_spectra()
            self._draw_conc()

    def _on_done(self, result: FitResult):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        self._result = result

        if result.aborted:
            self.status.setText("Stopped by user.")
            return
        if not result.success and not result.spectra:
            self.status.setText(f"Fit failed: {result.message}")
            QMessageBox.warning(self, "Fit failed",
                                result.message or "Unknown error.")
            return

        taus = self._format_lifetimes(result)
        self.status.setText(
            f"Done. redχ²={result.redchi:.3e}, nfev={result.nfev}. {taus}")
        self._refresh_results()

    @staticmethod
    def _format_lifetimes(result: FitResult) -> str:
        if result.params is None:
            return ""
        vals = []
        for key, par in result.params.items():
            if key.startswith("log_tau_"):
                vals.append(float(np.exp(par.value)))
        if not vals:
            return ""
        vals = sorted(vals)
        return "τ = " + ", ".join(f"{v:.3g}" for v in vals) + " ps"

    def closeEvent(self, event):
        # Make sure a running fit is stopped and the thread is joined before
        # the dialog is destroyed, so we never tear down a live QThread.
        if self._fitter is not None:
            self._fitter.abort()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)
