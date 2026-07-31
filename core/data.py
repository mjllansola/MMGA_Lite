"""TADataset: load and hold transient-absorption data (time x wavelength)."""

import logging
import re
import numpy as np
from pathlib import Path
from scipy.special import erfc


def _exp_irf(t: np.ndarray, tau: float, t0: float, fwhm: float) -> np.ndarray:
    """Convolution of H(t-t0)*exp(-(t-t0)/tau) with Gaussian IRF (fwhm)."""
    if fwhm <= 0:
        return np.where(t >= t0, np.exp(-(t - t0) / tau), 0.0)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    tt = t - t0
    exponent = np.clip(-tt / tau + sigma ** 2 / (2.0 * tau ** 2), -500, 500)
    result = 0.5 * np.exp(exponent) * erfc(
        -(tt - sigma ** 2 / tau) / (np.sqrt(2.0) * sigma)
    )
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


class TADataset:
    """
    Container for a 2-D transient-absorption dataset.

    Attributes
    ----------
    time : ndarray, shape (nt,)
    wavelength : ndarray, shape (nw,)
    data : ndarray, shape (nt, nw)   -- delta-OD matrix
    name : str
    """

    def __init__(self):
        self.time: np.ndarray = np.array([])
        self.wavelength: np.ndarray = np.array([])
        self.data: np.ndarray = np.array([])
        self.name: str = ""
        # Spectral-axis unit declared by the source file, e.g. "cm-1" or "nm".
        # None when the file does not record one; axis values are never converted.
        self.x_unit: str | None = None

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str) -> "TADataset":
        """
        Load from a whitespace- or comma-delimited text file.

        Supported layouts
        -----------------
        Layout DAT (typical TA export):
            0.0   491.2   492.9   ...       <- first numeric row: wavelengths
           -8.0   dOD     dOD     ...       <- subsequent rows: time + dOD

        Layout A (times in first column, no wavelength header):
            -1.0   0.00  0.01  ...

        Layout B (times in header, wavelengths in first column).
        """
        ds = cls()
        ds.name = Path(path).stem

        # Guard against accidentally loading a huge / wrong file.
        _MAX_BYTES = 500 * 1024 * 1024
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = 0
        if size > _MAX_BYTES:
            raise ValueError(
                f"File is too large to load ({size / 1e6:.0f} MB; limit "
                f"{_MAX_BYTES / 1e6:.0f} MB). Check that this is a data file.")

        # Count leading text (non-numeric) header lines to skip. Tokenising is
        # delimiter-agnostic so comma-separated files are handled correctly.
        n_skip = 0
        header_lines: list[str] = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    n_skip += 1
                    continue
                if stripped.startswith("#"):
                    header_lines.append(line)
                    n_skip += 1
                    continue
                tokens = [t for t in re.split(r"[,;\t\s]+", stripped) if t]
                if not tokens:
                    n_skip += 1
                    continue
                try:
                    float(tokens[0])
                    break   # first numeric line found
                except ValueError:
                    # First token is text. It may be a labelled data-header row
                    # (a text corner label followed by the numeric wavelength
                    # axis) -- detect it so we don't discard the axis.
                    numeric_rest = 0
                    for t in tokens[1:]:
                        try:
                            float(t)
                            numeric_rest += 1
                        except ValueError:
                            pass
                    if numeric_rest >= 3 and numeric_rest >= 0.5 * (len(tokens) - 1):
                        break
                    header_lines.append(line)
                    n_skip += 1

        # Read an explicit "x_unit=<unit>" tag if the writer recorded one.
        m_unit = re.search(r"x_unit\s*=\s*([^;,\s]+)", "\n".join(header_lines),
                           flags=re.IGNORECASE)
        if m_unit:
            ds.x_unit = m_unit.group(1).strip()

        # Sniff the delimiter from the numeric body of the file.
        delim = None
        try:
            import csv
            with open(path, encoding="utf-8", errors="replace") as f:
                for _ in range(n_skip):
                    f.readline()
                sample = f.read(4096)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
                delim = dialect.delimiter
            except csv.Error:
                first_line = sample.splitlines()[0] if sample else ""
                if "\t" in first_line:
                    delim = "\t"
                elif "," in first_line:
                    delim = ","
                elif ";" in first_line:
                    delim = ";"
                else:
                    delim = None
        except (OSError, UnicodeError):
            delim = None

        body = np.genfromtxt(path, delimiter=delim, skip_header=n_skip)

        if body.ndim < 2 or body.shape[0] < 2:
            raise ValueError("Could not parse numeric data from file.")

        # A labelled header row leaves body[0, 0] = NaN -- a cosmetic corner
        # label, not data. Replace it so the layout heuristic below works.
        if not np.isfinite(body[0, 0]):
            body[0, 0] = 0.0

        first_col = body[:, 0]
        matrix    = body[:, 1:]

        first_row_vals = matrix[0, :]
        if abs(float(first_col[0])) < 1e-9 and np.median(np.abs(first_row_vals)) > 100:
            # DAT layout: row 0 is the wavelength header, rows 1+ are data.
            ds.wavelength = first_row_vals
            ds.time       = first_col[1:]
            ds.data       = matrix[1:, :]
        elif np.median(np.abs(first_col)) > 100:
            # Layout B: wavelengths in first column, times across the header.
            ds.time       = first_row_vals
            ds.wavelength = first_col[1:]
            ds.data       = matrix[1:, :].T
        else:
            # Layout A: first column = times; wavelengths not in file.
            ds.time       = first_col
            ds.wavelength = np.arange(1, matrix.shape[1] + 1, dtype=float)
            ds.data       = matrix

        # Drop fully-empty (all-NaN) time rows and wavelength columns; a single
        # all-NaN row otherwise poisons every wavelength column for the fitter.
        ds.data = np.asarray(ds.data, dtype=float)
        if ds.data.ndim == 2 and ds.data.size:
            finite = np.isfinite(ds.data)
            row_ok = finite.any(axis=1)
            col_ok = finite.any(axis=0)
            if not row_ok.all() or not col_ok.all():
                ds.data       = ds.data[np.ix_(row_ok, col_ok)]
                ds.time       = np.asarray(ds.time, dtype=float)[row_ok]
                ds.wavelength = np.asarray(ds.wavelength, dtype=float)[col_ok]
                logging.getLogger(__name__).info(
                    "%s: dropped %d all-NaN row(s) and %d all-NaN column(s) "
                    "on load", ds.name, int((~row_ok).sum()), int((~col_ok).sum()))

        return ds

    @classmethod
    def from_numpy(cls, time: np.ndarray, wavelength: np.ndarray,
                   data: np.ndarray, name: str = "dataset") -> "TADataset":
        ds = cls()
        ds.time = np.asarray(time, dtype=float)
        ds.wavelength = np.asarray(wavelength, dtype=float)
        ds.data = np.asarray(data, dtype=float)
        ds.name = name
        return ds

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def nt(self) -> int:
        return len(self.time)

    @property
    def nw(self) -> int:
        return len(self.wavelength)

    def kinetic_trace(self, wavelength: float) -> np.ndarray:
        idx = np.argmin(np.abs(self.wavelength - wavelength))
        return self.data[:, idx]

    def spectral_slice(self, time: float) -> np.ndarray:
        idx = np.argmin(np.abs(self.time - time))
        return self.data[idx, :]

    # ------------------------------------------------------------------
    # Demo data generator (for quick testing without a data file)
    # ------------------------------------------------------------------

    @classmethod
    def make_demo(cls) -> "TADataset":
        """
        Synthetic TA dataset: 3-component parallel decay with IRF convolution.

        Parameters match the default parallel-model starting values so the
        demo converges cleanly: taus = [1, 20, 200] ps, t0 = 0, IRF = 0.15 ps.
        """
        rng = np.random.default_rng(42)
        t = np.concatenate([np.linspace(-0.5, 0.0, 25),
                            np.geomspace(0.01, 800.0, 220)])
        w = np.arange(450, 750, 5, dtype=float)

        t0 = 0.0
        fwhm = 0.15   # ps

        taus = [1.0, 20.0, 200.0]
        das = [
            np.exp(-0.5 * ((w - 520) / 30) ** 2),
            -0.5 * np.exp(-0.5 * ((w - 550) / 40) ** 2),
            0.3 * np.exp(-0.5 * ((w - 600) / 50) ** 2),
        ]
        data = np.zeros((len(t), len(w)))
        for tau, d in zip(taus, das):
            profile = _exp_irf(t, tau, t0, fwhm)
            data += np.outer(profile, d)

        data += rng.normal(0, 5e-4, data.shape)
        return cls.from_numpy(t, w, data,
                              name="demo_parallel (tau=1,20,200ps IRF=0.15ps)")

    @classmethod
    def make_raman_multi_demo(cls) -> "list[TADataset]":
        """Four-matrix FSRRS demo for multi-matrix global fitting.

        Real lycopene-in-THF FSRRS series: same sample and actinic pump
        (510 nm), four Raman-pump wavelengths (540, 550, 575, 590 nm). Sharing
        the kinetics across the four matrices in one global fit is the textbook
        use case. The x-axis is Raman shift (cm-1).

        Data source: Bercy, R.; D'mello, V. C.; Gall, A.; Ilioaia, C.; Pascal,
        A. A.; Romero, J. J.; Robert, B.; Llansola-Portoles, M. J., Reassessing
        Carotenoid Photophysics: Shedding Light on Dark States. J. Am. Chem.
        Soc. 2026, 148 (23), 23976-23985.

        Returns the datasets that are present in ``data/`` (an empty list if the
        bundled files are missing).
        """
        names = [
            "lyc_THF_AP510RP540.dat",
            "lyc_THF_AP510RP550.dat",
            "lyc_THF_AP510RP575.dat",
            "lyc_THF_AP510RP590.dat",
        ]
        base = Path(__file__).resolve().parent.parent / "data"
        out: list[TADataset] = []
        for fn in names:
            p = base / fn
            if not p.exists():
                continue
            ds = cls.from_csv(str(p))
            ds.name = fn.rsplit(".", 1)[0]
            ds.x_unit = "cm-1"        # axis is Raman shift, not wavelength
            out.append(ds)
        return out
