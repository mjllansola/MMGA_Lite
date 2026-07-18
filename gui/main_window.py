"""Main window: load transient-absorption matrices and explore them.

Deliberately minimal -- a matrix loader and a viewer with slide bars. There
are no data-processing controls. From here the user opens the global-fit
dialog to model the loaded matrices.
"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QFileDialog, QMessageBox,
    QSplitter, QAbstractItemView, QDialog,
)

from core.data import TADataset
from gui.plot_widgets import MatrixViewer
from gui.fit_dialog import FitDialog
from gui.chirp_dialog import ChirpDialog

_log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Global fitting — minimal")
        self.resize(1200, 760)

        self.datasets: list[TADataset] = []
        self._fit_dialog = None

        # ---- left column: matrix list + load buttons -------------------
        self.list = QListWidget()
        # Allow selecting several matrices at once so they can be removed
        # together (Ctrl/Shift-click).
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.currentRowChanged.connect(self._on_select)

        btn_open = QPushButton("Open matrices…")
        btn_open.clicked.connect(self._open_files)
        btn_demo = QPushButton("Load demo matrix")
        btn_demo.clicked.connect(self._load_demo)
        btn_raman = QPushButton("Load Raman demo (4 matrices)")
        btn_raman.setToolTip("Load the four lycopene FSRRS matrices "
                             "(Raman pump 540/550/575/590 nm) for a "
                             "multi-matrix global fit.")
        btn_raman.clicked.connect(self._load_raman_demo)
        btn_remove = QPushButton("Remove matrix")
        btn_remove.setToolTip("Remove the selected matrix(es). "
                              "Ctrl/Shift-click to select several.")
        btn_remove.clicked.connect(self._remove_selected)

        self.btn_chirp = QPushButton("Remove chirp…")
        self.btn_chirp.setToolTip("Manually correct group-velocity dispersion "
                                  "(chirp) on the selected matrix.")
        self.btn_chirp.clicked.connect(self._remove_chirp)
        self.btn_chirp.setEnabled(False)

        self.btn_fit = QPushButton("Global fit…")
        self.btn_fit.clicked.connect(self._open_fit)
        self.btn_fit.setEnabled(False)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Loaded matrices"))
        ll.addWidget(self.list, 1)
        ll.addWidget(btn_open)
        ll.addWidget(btn_demo)
        ll.addWidget(btn_raman)
        ll.addWidget(btn_remove)
        ll.addSpacing(12)
        ll.addWidget(self.btn_chirp)
        ll.addWidget(self.btn_fit)

        # ---- right: viewer --------------------------------------------
        self.viewer = MatrixViewer()

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.viewer)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 940])
        self.setCentralWidget(split)

        self.statusBar().showMessage("Open one or more TA matrices to begin.")

    # ------------------------------------------------------------------

    def _open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open TA matrix files", "",
            "Data files (*.dat *.csv *.txt *.tsv);;All files (*)")
        loaded = 0
        for p in paths:
            try:
                ds = TADataset.from_csv(p)
            except Exception as exc:
                QMessageBox.warning(self, "Load failed",
                                    f"Could not load:\n{p}\n\n{exc}")
                continue
            self.datasets.append(ds)
            self.list.addItem(QListWidgetItem(ds.name))
            loaded += 1
        if loaded:
            self.list.setCurrentRow(self.list.count() - 1)
            self._refresh_state()
            self.statusBar().showMessage(f"Loaded {loaded} matrix(es).")

    def _load_demo(self):
        ds = TADataset.make_demo()
        self.datasets.append(ds)
        self.list.addItem(QListWidgetItem(ds.name))
        self.list.setCurrentRow(self.list.count() - 1)
        self._refresh_state()
        self.statusBar().showMessage("Loaded synthetic demo matrix.")

    def _load_raman_demo(self):
        matrices = TADataset.make_raman_multi_demo()
        if not matrices:
            QMessageBox.warning(
                self, "Raman demo unavailable",
                "The bundled Raman demo files were not found in the data "
                "folder.")
            return
        for ds in matrices:
            self.datasets.append(ds)
            self.list.addItem(QListWidgetItem(ds.name))
        self.list.setCurrentRow(self.list.count() - 1)
        self._refresh_state()
        self.statusBar().showMessage(
            f"Loaded {len(matrices)} Raman matrices — open Global fit… to try "
            "the multi-matrix fit.")

    def _remove_selected(self):
        # Collect every selected row (fall back to the current row), then
        # remove from highest index down so earlier indices stay valid.
        rows = sorted({self.list.row(it) for it in self.list.selectedItems()},
                      reverse=True)
        if not rows:
            r = self.list.currentRow()
            if r < 0:
                return
            rows = [r]
        # Mutate the model with list signals blocked so currentRowChanged does
        # not fire against a half-updated dataset list; refresh the view after.
        self.list.blockSignals(True)
        for r in rows:
            self.list.takeItem(r)
            del self.datasets[r]
        self.list.blockSignals(False)
        self._refresh_state()
        if self.datasets:
            new_row = min(rows[-1], self.list.count() - 1)
            self.list.setCurrentRow(new_row)
            self.viewer.set_dataset(self.datasets[new_row])
        else:
            self.list.setCurrentRow(-1)
            self.viewer.set_dataset(None)
        self.statusBar().showMessage(f"Removed {len(rows)} matrix(es).")

    def _on_select(self, row):
        if 0 <= row < len(self.datasets):
            self.viewer.set_dataset(self.datasets[row])

    def _refresh_state(self):
        has = len(self.datasets) > 0
        self.btn_fit.setEnabled(has)
        self.btn_chirp.setEnabled(has)

    def _remove_chirp(self):
        row = self.list.currentRow()
        if not (0 <= row < len(self.datasets)):
            return
        ds = self.datasets[row]
        x_label = f"Axis ({ds.x_unit})" if ds.x_unit else "Wavelength (nm)"
        dlg = ChirpDialog(ds, x_label=x_label, parent=self)
        if dlg.exec() == QDialog.Accepted:
            corrected = dlg.corrected_dataset()
            if corrected is not None:
                # Replace the matrix in place with the chirp-corrected version.
                self.datasets[row] = corrected
                self.viewer.set_dataset(corrected)
                self.statusBar().showMessage(
                    f"Chirp correction applied to '{corrected.name}'.")

    def _open_fit(self):
        if not self.datasets:
            return
        # A fresh dialog each time keeps state simple and predictable.
        self._fit_dialog = FitDialog(list(self.datasets), parent=self)
        self._fit_dialog.show()
