"""Entry point for the minimal global-fitting application.

Global and target analysis of transient-absorption data by variable
projection. Load one or more time x wavelength matrices, explore them with
the slide bars, then fit a kinetic model shared across all of them.
"""

import argparse
import logging
import os
import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow

__version__ = "0.1.0"


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="global-fitting",
        description="Global / target analysis viewer for "
                    "transient-absorption data.",
    )
    parser.add_argument("path", nargs="?", default=None,
                        help="Optional data file to open on startup.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO).")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser.parse_known_args(argv)


def main():
    args, qt_args = _parse_args(sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    app = QApplication([sys.argv[0]] + qt_args)
    app.setStyle("Fusion")
    win = MainWindow()
    win.showMaximized()

    if args.path:
        if os.path.isfile(args.path):
            try:
                from core.data import TADataset
                ds = TADataset.from_csv(args.path)
                win.datasets.append(ds)
                win.list.addItem(ds.name)
                win.list.setCurrentRow(win.list.count() - 1)
                win._refresh_state()
            except Exception:
                logging.getLogger(__name__).exception(
                    "Could not open startup file %s", args.path)
        else:
            logging.getLogger(__name__).warning(
                "Startup path does not exist: %s", args.path)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
