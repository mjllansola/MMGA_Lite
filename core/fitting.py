"""
GlobalFitter: variable-projection global fitting engine.
Fits one or more datasets simultaneously with shared kinetic parameters;
the associated spectra are solved independently per dataset.

Strategy
--------
For given nonlinear parameters theta (lifetimes / rates / t0 / IRF):
  1. For each dataset i:
     a. Compute concentration profiles  C_i(t_i; theta)   shape (nt_i, nc)
     b. Solve  Data_i ~ C_i @ SAS_i  by lstsq / nnls  -> SAS_i
     c. Residual_i = Data_i - C_i @ SAS_i
  2. Concatenate all residuals and hand to lmfit.

Key numerical choices:
  - Lifetimes parametrised as log(tau) in the models (see models.py).
  - ``x_scale='jac'`` passed to scipy least_squares for automatic
    Jacobian-based trust-region scaling.
  - Residuals optionally normalised per wavelength column so that
    low-signal spectral regions are not silenced by high-signal ones.

Provenance
----------
The variable-projection separation of linear (SAS) and nonlinear (kinetic)
parameters is due to Golub & Pereyra (1973), SIAM J. Numer. Anal. 10,
413-432. Its application to time-resolved spectroscopy follows van Stokkum,
Larsen & van Grondelle (2004), BBA Bioenergetics 1657, 82-104. The
implementation in this file is independent and original.
"""

import logging

import numpy as np
from scipy.optimize import nnls
import lmfit

from .data import TADataset
from .models import BaseModel

_log = logging.getLogger(__name__)


def _drop_nan_rows(A: np.ndarray, B: np.ndarray) -> tuple:
    """Restrict a linear system ``A x ~ B`` to the rows where *B* is fully
    finite, so a few NaN data rows don't poison the whole batched solve.

    A single NaN anywhere in *B* propagates through lstsq / NNLS and turns
    every solved spectrum point into NaN. Dropping the offending data rows
    lets the solve use the finite delays; the residual on the excluded rows
    stays NaN and is omitted downstream. Fast-paths the all-finite case.
    """
    if B.size and not np.isfinite(B).all():
        ok = np.isfinite(B).all(axis=1)
        if ok.any() and not ok.all():
            return A[ok], B[ok]
    return A, B


def _nnls_multi(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve  min_{x>=0} ||A x - b||  for every column b of B.

    Mathematically identical to looping ``scipy.optimize.nnls`` over the
    columns of B, but when A is tall (nt >> nc, the usual TA case) we first
    reduce A to its thin-QR factor R (nc x nc) and project every right-hand
    side with Q^T. NNLS is invariant under this orthogonal transform, so each
    per-column solve runs on an nc x nc system instead of an nt x nc one.
    Each reduced solve falls back to the full-matrix NNLS if it raises.
    """
    n_rhs = B.shape[1]
    if n_rhs == 0:
        return np.zeros((A.shape[1], 0))
    A, B = _drop_nan_rows(A, B)
    n_rows, n_cols = A.shape
    X = np.zeros((n_cols, n_rhs))
    if n_rows > n_cols:
        Q, R = np.linalg.qr(A)
        Rhs = Q.T @ B
        for j in range(n_rhs):
            try:
                X[:, j], _ = nnls(R, Rhs[:, j])
            except Exception as exc:
                _log.debug("NNLS QR-reduced solve failed (col %d), "
                           "falling back to full NNLS: %s", j, exc)
                X[:, j], _ = nnls(A, B[:, j])
    else:
        for j in range(n_rhs):
            X[:, j], _ = nnls(A, B[:, j])
    return X


def transform_spectra(X: np.ndarray, kind: str, spectral_type: str,
                      M_real, taus_real, labels) -> np.ndarray | None:
    """Convert as-fitted spectra *X* (nc x nw) to the requested view.

    Parameters
    ----------
    X : ndarray, shape (nc, nw)
        As-fitted spectral matrix (SAS when spectral_type='SAS', or DAS
        when spectral_type='DAS').
    kind : str
        One of 'SAS', 'DAS', or 'EADS' (case-insensitive).
    spectral_type : str
        The native type of *X*: 'SAS' or 'DAS' (from the model).
    M_real : ndarray | None
        Eigenmode matrix (n x n) satisfying DAS = M_real @ SAS. None -> identity.
    taus_real : list[float] | None
        Lifetimes (ps) matching the rows/cols of M_real.
    labels : list[str]
        Per-component labels.

    Returns
    -------
    ndarray of shape (nc_view, nw), or None if X is None/empty.
    """
    if X is None or X.size == 0:
        return None

    k = (kind or '').upper()
    if k == 'EDAS':
        k = 'EADS'
    stype = (spectral_type or 'SAS').upper()

    nc = X.shape[0]

    if stype == 'DAS':
        DAS = X
        SAS = X
    else:
        SAS = X
        if M_real is not None:
            M = np.asarray(M_real, dtype=float)
            if M.shape == (nc, nc):
                DAS = M @ X
            else:
                DAS = X
        else:
            DAS = X

    if k == 'SAS':
        return SAS
    if k == 'DAS':
        return DAS

    if k == 'EADS':
        if stype == 'SAS':
            # A sequential scheme's species spectra ARE the EADS by convention.
            return SAS
        # Native DAS -> build EADS generically: cumulative sum of DAS rows
        # ordered from shortest to longest tau.
        real_taus = list(taus_real) if taus_real is not None else []
        if len(real_taus) == nc and real_taus:
            order = list(np.argsort(np.asarray(real_taus)))
        else:
            order = list(range(nc))
        ordered = DAS[order, :]
        return np.cumsum(ordered, axis=0)

    return X


class FitResult:
    def __init__(self):
        self.params: lmfit.Parameters | None = None
        # Per-dataset results
        self.spectra:   list[np.ndarray] = []   # each (nc, nw_i)
        self.profiles:  list[np.ndarray] = []   # each (nt_i, nc)
        self.residuals: list[np.ndarray] = []   # each (nt_i, nw_i)
        self.datasets:  list[TADataset]  = []
        self.chisqr:  float = np.inf
        self.redchi:  float = np.inf
        self.nfev:    int   = 0
        self.message: str   = ''
        self.success: bool  = False
        # True when the run was stopped via GlobalFitter.abort().
        self.aborted: bool  = False
        self.labels:       list[str] = []
        # Algebraic native type of the as-fitted spectra ('SAS' or 'DAS').
        self.spectral_type: str = 'SAS'
        # Conventional display nomenclature: 'DAS'/'EADS'/'SAS'.
        self.display_type: str = ''
        # Per-dataset eigenmode conversion matrices for SAS<->DAS transforms.
        self.eigenmode_M:    list = []
        self.eigenmode_taus: list = []
        # Provenance for export.
        self.model_name:  str  = ''
        self.nonneg:      bool = False
        self.fit_method:  str  = ''
        self.auto_weight: bool = True

    # convenience: first dataset results
    @property
    def spectra_0(self) -> np.ndarray | None:
        return self.spectra[0] if self.spectra else None

    @property
    def profiles_0(self) -> np.ndarray | None:
        return self.profiles[0] if self.profiles else None

    @property
    def residuals_0(self) -> np.ndarray | None:
        return self.residuals[0] if self.residuals else None

    def display_nomenclature(self) -> str:
        """User-facing spectral nomenclature: 'DAS'/'EADS'/'SAS'."""
        d = (self.display_type or '').upper()
        if d:
            return d
        st = (self.spectral_type or 'SAS').upper()
        if st == 'DAS':
            return 'DAS'
        if 'sequential' in (self.model_name or '').lower():
            return 'EADS'
        return 'SAS'

    def fitted_data(self, idx: int = 0) -> np.ndarray | None:
        if self.profiles and self.spectra and idx < len(self.profiles):
            return self.profiles[idx] @ self.spectra[idx]
        return None


class GlobalFitter:
    """
    Parameters
    ----------
    datasets : TADataset or list[TADataset]
        One or more datasets. Kinetic parameters are shared; the associated
        spectra are solved independently per dataset.
    model : BaseModel
    nonneg : bool
        Constrain the associated spectra to be non-negative (NNLS solve).
    method : str
        lmfit minimiser name (default 'least_squares').
    auto_weight : bool
        Normalise residuals per wavelength column (1/max|Data[:,j]|, capped at
        4x) so weak spectral channels are not silenced by strong ones.
    """

    def __init__(self, datasets, model: BaseModel,
                 nonneg: bool = False, method: str = 'least_squares',
                 max_nfev: int = 10000,
                 tol: float = 1e-7,
                 auto_weight: bool = True):
        if isinstance(datasets, TADataset):
            datasets = [datasets]
        self.datasets  = datasets
        self.model     = model
        self.nonneg    = nonneg
        self.method    = method
        self.max_nfev  = max_nfev
        self.tol       = tol
        self.auto_weight = auto_weight
        self._abort    = False
        # Per-dataset per-wavelength column scales, precomputed in run().
        self._col_scales: list[np.ndarray | None] = []

    def abort(self):
        self._abort = True

    # ------------------------------------------------------------------

    @staticmethod
    def _col_scale_for_data(D: np.ndarray) -> 'np.ndarray | None':
        """Return 1/max(|D[:,j]|) with a 25% floor (amplification <= 4x), or
        None on degenerate input. Caps how much a weak / near-isosbestic
        column can be up-weighted so noise-only columns can't dominate chi^2.
        """
        col_max = np.max(np.abs(D), axis=0)
        overall_max = float(np.max(col_max)) if col_max.size > 0 else 1.0
        if not np.isfinite(overall_max) or overall_max <= 0:
            return None
        floor = overall_max * 0.25
        col_max = np.where(col_max < floor, floor, col_max)
        return 1.0 / col_max

    def _solve(self, C: np.ndarray, Data: np.ndarray) -> np.ndarray:
        """Solve Data ~ C @ X for the associated spectra X (nc x nw).

        NaN-robust: restricts to finite data rows so a few NaN edge delays
        don't turn every spectrum point into NaN.
        """
        if self.nonneg:
            return _nnls_multi(C, Data)
        C_solve, D_solve = _drop_nan_rows(C, Data)
        # rcond truncates singular values below rcond*max(sv), stabilising the
        # solution when C is ill-conditioned (e.g. two similar lifetimes).
        X, _, _, _ = np.linalg.lstsq(C_solve, D_solve, rcond=1e-10)
        return X

    def _params_for_dataset(self, params: lmfit.Parameters,
                            idx: int) -> lmfit.Parameters:
        """Return a params copy with t0 set to a per-dataset value if present."""
        key = f't0_{idx}'
        if key not in params:
            return params
        p = params.copy()
        p['t0'].set(value=params[key].value)
        return p

    def _residuals(self, params: lmfit.Parameters) -> np.ndarray:
        if self._abort:
            return np.zeros(sum(ds.data.size for ds in self.datasets))
        parts = []
        for i, ds in enumerate(self.datasets):
            p = self._params_for_dataset(params, i)
            C = self.model.concentration_profiles(ds.time, p)
            X = self._solve(C, ds.data)
            resid = ds.data - C @ X
            if self._col_scales and i < len(self._col_scales):
                s = self._col_scales[i]
                if s is not None:
                    resid = resid * s[np.newaxis, :]
            parts.append(resid.ravel())
        return np.concatenate(parts)

    # ------------------------------------------------------------------

    def run(self, params: lmfit.Parameters,
            progress_cb=None) -> FitResult:
        self._abort = False
        result = FitResult()
        result.datasets = self.datasets
        result.spectral_type = self.model.spectral_type()
        result.display_type = self.model.display_spectral_type()
        result.model_name = str(getattr(self.model, 'name', type(self.model).__name__))
        result.nonneg      = bool(self.nonneg)
        result.fit_method  = str(self.method)
        result.auto_weight = bool(self.auto_weight)

        n_key   = 'n_exp' if 'n_exp' in params else 'n_comp'
        n_comp  = int(params[n_key].value) if n_key in params else 2
        result.labels = self.model.component_labels(n_comp)

        # Precompute per-wavelength auto-weight column scales.
        if self.auto_weight:
            self._col_scales = [self._col_scale_for_data(ds.data)
                                for ds in self.datasets]
        else:
            self._col_scales = [None] * len(self.datasets)

        iter_count = [0]

        def _cb(params, it, resid, *a, **kw):
            iter_count[0] += 1
            if progress_cb:
                progress_cb(iter_count[0], params, resid)
            return self._abort or None

        # Pre-flight: if every residual point is empty or NaN, lmfit with
        # nan_policy='omit' would "converge" instantly on an empty problem
        # (chi2=0, success=True, blank tabs). Fail with a diagnosis instead.
        try:
            r0 = np.asarray(self._residuals(params), dtype=float).ravel()
        except Exception:
            r0 = None
        if r0 is not None:
            finite = np.isfinite(r0)
            if r0.size == 0 or not finite.any():
                result.message = (
                    'nothing to fit: every residual point is empty or NaN '
                    '- check for empty matrices or all-NaN data')
                return result
            if float(np.max(np.abs(r0[finite]))) == 0.0:
                result.message = (
                    'nothing to fit: all residuals are exactly zero at the '
                    'starting parameters - the data is zero')
                return result

        try:
            if self.method == 'least_squares':
                tol_kw = dict(ftol=self.tol, xtol=self.tol, gtol=self.tol,
                              x_scale='jac')
            else:
                tol_kw = dict(tol=self.tol)
            lm = lmfit.minimize(
                self._residuals, params,
                method=self.method, iter_cb=_cb, nan_policy='omit',
                max_nfev=self.max_nfev,
                **tol_kw,
            )
        except Exception as exc:
            result.message = str(exc)
            return result

        if self._abort:
            result.aborted = True
            result.success = False
            result.nfev    = int(getattr(lm, 'nfev', 0) or 0)
            result.message = 'stopped by user'
            return result

        result.params   = lm.params
        result.chisqr   = lm.chisqr
        result.redchi   = lm.redchi
        result.nfev     = lm.nfev
        result.message  = lm.message
        result.success  = lm.success

        # Recompute per-dataset results at the best parameters.
        for i, ds in enumerate(self.datasets):
            p = self._params_for_dataset(lm.params, i)
            C = self.model.concentration_profiles(ds.time, p)
            X = self._solve(C, ds.data)
            result.profiles.append(C)
            result.spectra.append(X)
            result.residuals.append(ds.data - C @ X)

            # Store eigenmode conversion matrix (SAS->DAS) for the spectra views.
            try:
                em = self.model.eigenmode_matrix(p)
                if em is not None:
                    result.eigenmode_M.append(em[0])
                    result.eigenmode_taus.append(em[1])
                else:
                    result.eigenmode_M.append(None)
                    result.eigenmode_taus.append(None)
            except Exception:
                result.eigenmode_M.append(None)
                result.eigenmode_taus.append(None)

        return result

    def evaluate_at(self, params: lmfit.Parameters) -> "FitResult":
        """Compute spectra/profiles/residuals at *params* without optimising.

        Used for the live preview during iteration (called from the progress
        callback on the worker thread).
        """
        result = FitResult()
        result.datasets = self.datasets
        result.spectral_type = self.model.spectral_type()
        result.display_type = self.model.display_spectral_type()
        result.model_name = str(getattr(self.model, 'name', type(self.model).__name__))
        result.nonneg      = bool(self.nonneg)
        result.fit_method  = str(self.method)
        n_key = "n_exp" if "n_exp" in params else "n_comp"
        n_comp = int(params[n_key].value) if n_key in params else 2
        result.labels = self.model.component_labels(n_comp)

        for i, ds in enumerate(self.datasets):
            p = self._params_for_dataset(params, i)
            C = self.model.concentration_profiles(ds.time, p)
            try:
                X = self._solve(C, ds.data)
                result.profiles.append(C)
                result.spectra.append(X)
                result.residuals.append(ds.data - C @ X)
            except Exception:
                result.profiles.append(C)
                result.spectra.append(np.zeros((C.shape[1], ds.nw)))
                result.residuals.append(ds.data.copy())
        result.params = params
        return result
