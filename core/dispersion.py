"""Chirp (group-velocity dispersion) correction.

The user marks a few (wavelength, t0) control points by hand; a polynomial is
fitted through them and every wavelength column is shifted in time so that all
columns share a common time zero. There is no automatic guessing here -- the
control points come only from manual placement.
"""

import numpy as np


def fit_chirp_poly(points, order: int) -> np.ndarray:
    """Least-squares polynomial t0(wavelength) through manually placed points.

    Parameters
    ----------
    points : sequence of (wavelength, t0)
    order : int
        Requested polynomial order; clamped to ``len(points) - 1`` so the fit
        is never under-determined.

    Returns the polynomial coefficients (highest power first, as for
    ``np.polyval``). Raises ValueError if fewer than two points are given.
    """
    pts = list(points)
    if len(pts) < 2:
        raise ValueError("Need at least two control points to fit a chirp curve.")
    wls = np.array([p[0] for p in pts], dtype=float)
    t0s = np.array([p[1] for p in pts], dtype=float)
    order = int(max(1, min(order, len(pts) - 1)))
    return np.polyfit(wls, t0s, order)


def apply_dispersion_correction(data: np.ndarray,
                                time: np.ndarray,
                                wavelength: np.ndarray,
                                poly_coeffs: np.ndarray) -> np.ndarray:
    """Remove chirp from a TA matrix given a polynomial time-zero curve.

    For each wavelength column *i* the group-delay offset is
        t0_i = np.polyval(poly_coeffs, wavelength[i])
    and the column is resampled at ``time + t0_i`` so its onset moves to t = 0.
    Points shifted outside the measured range are clamped to the column's
    first/last sample (never zero-filled, which would add a sharp edge
    artifact). The input matrix is not modified; a new array is returned.
    """
    data = np.asarray(data, dtype=float)
    time = np.asarray(time, dtype=float)
    corrected = np.zeros_like(data)
    for i, wl in enumerate(wavelength):
        t0 = float(np.polyval(poly_coeffs, wl))
        col = data[:, i]
        corrected[:, i] = np.interp(time + t0, time, col,
                                    left=col[0], right=col[-1])
    return corrected
