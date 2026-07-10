"""
Kinetic models for global TA analysis.

Each model exposes:
    concentration_profiles(t, params) -> C  shape (nt, n_components)
    default_params(n) -> lmfit.Parameters

The fitter uses variable projection:
    given C(t), solve  Data = C @ SAS  in least squares sense per wavelength.

NOTE: All lifetime parameters are stored and optimised in log-space
(parameter name ``log_tau_*``, value = ln(tau/ps)).  This is critical for
convergence when lifetimes span orders of magnitude (0.1 ps - 1000 ps)
because it equalises the parameter scales and keeps the Jacobian
well-conditioned.  Models recover tau via ``exp(log_tau)``.

Provenance
----------
The compartmental target-analysis formalism, the column-based K matrix,
and the analytical exp-Gaussian (erfc) IRF convolution formulas used in
this module are taken from van Stokkum, Larsen & van Grondelle (2004),
BBA Bioenergetics 1657, 82-104 (doi:10.1016/j.bbabio.2004.04.011).
The implementation is an original NumPy / SciPy reproduction of those
published formulas.
"""

import numpy as np
from scipy.special import erfc
from scipy.linalg import eig
import lmfit


# ---------------------------------------------------------------------------
# IRF convolution helpers (single Gaussian)
# ---------------------------------------------------------------------------

def _exp_irf_analytic(t: np.ndarray, tau: float,
                      t0: float, fwhm: float) -> np.ndarray:
    """
    Exact convolution of H(t-t0)*exp(-(t-t0)/tau) with a Gaussian IRF.

    Analytic formula (derived from completing the square):
      C(t) = 0.5 * exp(-tt/tau + sigma^2/(2*tau^2))
               * erfc( -(tt - sigma^2/tau) / (sqrt(2)*sigma) )
    where tt = t - t0, sigma = FWHM / (2*sqrt(2*ln2)).
    """
    if fwhm <= 0:
        return np.where(t >= t0, np.exp(-(t - t0) / tau), 0.0)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    tt = t - t0
    exponent = np.clip(-tt / tau + sigma ** 2 / (2.0 * tau ** 2), -500, 500)
    result = 0.5 * np.exp(exponent)
    result *= erfc(-(tt - sigma ** 2 / tau) / (np.sqrt(2.0) * sigma))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _step_irf_analytic(t: np.ndarray, t0: float, fwhm: float) -> np.ndarray:
    """
    Exact convolution of H(t-t0) with a Gaussian IRF.
    Used for zero-rate (non-decaying) eigenmodes.
    """
    if fwhm <= 0:
        return np.where(t >= t0, 1.0, 0.0).astype(float)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    tt = t - t0
    return 0.5 * erfc(-tt / (np.sqrt(2.0) * sigma))


def _profiles_from_K_analytic(K: np.ndarray, init: np.ndarray,
                              t: np.ndarray,
                              t0: float, fwhm: float) -> np.ndarray:
    """
    Concentration profiles for dC/dt = K @ C, C(t0+) = init,
    analytically IRF-convolved.

    Method: eigendecompose K, analytically convolve each mode:
      C_conv(t) = sum_i  b_i * [exp(lambda_i * t) (x) IRF(t-t0)] * v_i
    where lambda_i are eigenvalues (<= 0 for a stable system).

    Uses scipy.linalg.eig and pseudoinverse pinv(V) instead of inv(V) to
    handle near-degenerate eigenvalues (e.g. two similar lifetimes).
    NOTE: do NOT clip to zero here -- clipping creates discontinuities in
    the parameter landscape that break gradient-based optimizers.
    """
    n = K.shape[0]
    nt = len(t)

    eigenvalues, V = eig(K)
    eigenvalues = eigenvalues.real
    V = V.real
    V_inv = np.linalg.pinv(V)
    b = V_inv @ init   # coefficients in eigenmode basis

    C = np.zeros((nt, n))
    for mode_idx in range(n):
        lam = eigenvalues[mode_idx]
        coeff = float(b[mode_idx])
        if abs(coeff) < 1e-14:
            continue
        if abs(lam) < 1e-10:
            # Zero eigenvalue -> non-decaying mode (Heaviside (x) IRF)
            mode_profile = _step_irf_analytic(t, t0, fwhm)
        elif lam < 0:
            tau = -1.0 / lam   # positive lifetime
            mode_profile = _exp_irf_analytic(t, tau, t0, fwhm)
        else:
            # Growing eigenvalue -- unphysical; smooth differentiable penalty.
            tau = 1.0 / (1.0 + lam)
            mode_profile = _exp_irf_analytic(t, tau, t0, fwhm)
        C += coeff * np.outer(mode_profile, V[:, mode_idx])

    return C


def _eigenmode_matrix_from_K(K: np.ndarray, init: np.ndarray):
    """Compute (M_out, lifetimes) from rate matrix K and initial population.

    M_out satisfies  DAS = M_out @ SAS  for SAS-fitted models.
    Derivation: C = E @ M_out.T with M_out = diag(b) @ V.T, b = V^-1 @ init.
    Returns (M_out, taus) where taus[k] = -1/lambda_k (positive lifetimes).
    """
    eigenvalues, V = eig(K)
    eigenvalues = eigenvalues.real
    V = V.real
    V_inv = np.linalg.pinv(V)
    b = V_inv @ init

    taus = []
    for lam in eigenvalues:
        if abs(lam) < 1e-10:
            taus.append(1e12)               # non-decaying mode
        elif lam < 0:
            taus.append(-1.0 / lam)
        else:
            taus.append(1.0 / (1.0 + lam))  # unphysical growing mode

    M_out = np.diag(b) @ V.T               # DAS = M_out @ SAS
    return (M_out, taus)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseModel:
    name: str = "base"
    n_components: int = 0

    def concentration_profiles(self, t, params) -> np.ndarray:
        raise NotImplementedError

    def default_params(self, n: int) -> lmfit.Parameters:
        raise NotImplementedError

    def component_labels(self, n: int) -> list[str]:
        return [f"C{i+1}" for i in range(n)]

    def spectral_type(self) -> str:
        """Algebraic native type of the fitted spectra: 'DAS' or 'SAS'."""
        return "SAS"

    def display_spectral_type(self) -> str:
        """Conventional nomenclature shown to the user.

        Parallel -> 'DAS', Sequential -> 'EADS', Target -> 'SAS'.
        """
        return self.spectral_type()

    def eigenmode_matrix(self, params: lmfit.Parameters):
        """Return (M_out, lifetimes) so DAS = M_out @ SAS, or None."""
        return None


# ---------------------------------------------------------------------------
# Parallel decay  (sum of exponentials)  -> DAS
# ---------------------------------------------------------------------------

class ParallelModel(BaseModel):
    """
    Psi(lambda,t) = sum_i DAS_i(lambda) * [exp(-t/tau_i) (x) IRF]

    Concentration profiles are the (IRF-convolved) exponentials; the fit
    recovers DAS as the linear coefficients. Lifetimes are parametrised as
    log_tau_i = ln(tau_i) for optimiser stability across orders of magnitude.
    """
    name = "Parallel (sum-of-exp)"

    def concentration_profiles(self, t: np.ndarray,
                               params: lmfit.Parameters) -> np.ndarray:
        t0 = params["t0"].value
        fwhm = params["irf_fwhm"].value
        n = int(params["n_exp"].value)
        C = np.zeros((len(t), n))
        for i in range(n):
            tau = np.exp(params[f"log_tau_{i+1}"].value)
            C[:, i] = _exp_irf_analytic(t, tau, t0, fwhm)
        return C

    def default_params(self, n: int) -> lmfit.Parameters:
        p = lmfit.Parameters()
        p.add("n_exp", value=n, vary=False)
        p.add("t0", value=0.0, min=-5.0, max=5.0)
        p.add("irf_fwhm", value=0.15, min=0.10, max=0.20, vary=False)
        if n <= 1:
            taus = np.array([10.0])
        else:
            taus = np.logspace(np.log10(0.5), np.log10(500.0), n)
        for i in range(n):
            tau = float(taus[i])
            p.add(f"log_tau_{i+1}", value=np.log(tau),
                  min=np.log(0.1), max=np.log(100000.0))
        return p

    def component_labels(self, n: int) -> list[str]:
        return [f"tau{i+1}" for i in range(n)]

    def spectral_type(self) -> str:
        return "DAS"

    def eigenmode_matrix(self, params: lmfit.Parameters):
        """Parallel model: each component IS its own eigenmode -> M_out = I."""
        n = int(params["n_exp"].value)
        taus = [float(np.exp(params[f"log_tau_{i+1}"].value)) for i in range(n)]
        return (np.eye(n), taus)


# ---------------------------------------------------------------------------
# Sequential model  A -> B -> C -> ...  -> ground state
# ---------------------------------------------------------------------------

class SequentialModel(BaseModel):
    """
    dA/dt = -k1*A
    dB/dt =  k1*A - k2*B
    ...
    C(t) convolved with IRF  -> SAS from linear LS.
    Lifetimes parametrised as log_tau_i = ln(tau_i).
    """
    name = "Sequential (A->B->C...)"

    def concentration_profiles(self, t: np.ndarray,
                               params: lmfit.Parameters) -> np.ndarray:
        t0   = params["t0"].value
        fwhm = params["irf_fwhm"].value
        n    = int(params["n_comp"].value)

        rates = np.array([1.0 / np.exp(params[f"log_tau_{i+1}"].value)
                          for i in range(n)])
        K = np.diag(-rates) + np.diag(rates[:-1], -1)

        init = np.zeros(n);  init[0] = 1.0
        return _profiles_from_K_analytic(K, init, t, t0, fwhm)

    def default_params(self, n: int) -> lmfit.Parameters:
        p = lmfit.Parameters()
        p.add("n_comp", value=n, vary=False)
        p.add("t0", value=0.0, min=-5.0, max=5.0)
        p.add("irf_fwhm", value=0.15, min=0.10, max=0.20, vary=False)
        if n <= 1:
            taus = np.array([10.0])
        else:
            taus = np.logspace(np.log10(0.5), np.log10(500.0), n)
        for i in range(n):
            tau = float(taus[i])
            p.add(f"log_tau_{i+1}", value=np.log(tau),
                  min=np.log(0.1), max=np.log(100000.0))
        return p

    def component_labels(self, n: int) -> list[str]:
        return [chr(65 + i) for i in range(n)]

    def spectral_type(self) -> str:
        return "SAS"

    def display_spectral_type(self) -> str:
        # Species spectra of a sequential scheme are conventionally EADS.
        return "EADS"

    def eigenmode_matrix(self, params: lmfit.Parameters):
        n = int(params["n_comp"].value)
        if n < 1:
            return None
        try:
            rates = np.array([1.0 / np.exp(params[f"log_tau_{i+1}"].value)
                              for i in range(n)])
            K = np.diag(-rates) + np.diag(rates[:-1], -1)
            init = np.zeros(n); init[0] = 1.0
            return _eigenmode_matrix_from_K(K, init)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Target / compartment model  (arbitrary K matrix)
# ---------------------------------------------------------------------------

class TargetModel(BaseModel):
    """
    User supplies the connectivity of an n-compartment scheme. Off-diagonal
    element K[i,j] is the rate from compartment j -> i; the diagonal is
    -(sum of out-rates). A self-connection (j -> j) encodes decay to the
    untracked ground state.
    """
    name = "Target (compartment)"

    def __init__(self):
        self._k_connections: list[tuple[int, int]] = []  # (to, from) pairs
        self._n: int = 2
        self._init: np.ndarray | None = None  # initial population vector

    def setup(self, n: int, connections: list[tuple[int, int]],
              init: np.ndarray | None = None):
        self._n = n
        self._k_connections = connections
        self._init = init if init is not None else np.eye(n)[:, 0]

    @staticmethod
    def _tau_key(i: int, j: int) -> str:
        """Parameter key for the connection (i <- j), i.e. transfer j -> i
        (both 0-based). Written source-first so the label reads in the
        physical direction of the arrow. A self-connection (i == j) encodes
        decay to the ground state and uses a 'G' suffix.
        """
        if i == j:
            return f"log_tau_{j+1}G"
        return f"log_tau_{j+1}{i+1}"

    def _build_K(self, params: lmfit.Parameters) -> np.ndarray:
        n = self._n
        K = np.zeros((n, n))
        # Ground-state losses (self-connections) are kept separate from the
        # off-diagonal column-conservation sum, otherwise the diagonal formula
        # cancels them and the system becomes closed (curves plateau).
        ground_loss = np.zeros(n)
        for (i, j) in self._k_connections:
            tau = np.exp(params[self._tau_key(i, j)].value)
            rate = 1.0 / tau
            if i == j:
                ground_loss[j] += rate      # decay to ground
            else:
                K[i, j] = rate              # transfer j -> i
        for col in range(n):
            K[col, col] = -(np.sum(K[:, col]) + ground_loss[col])
        return K

    def concentration_profiles(self, t: np.ndarray,
                               params: lmfit.Parameters) -> np.ndarray:
        t0   = params["t0"].value
        fwhm = params["irf_fwhm"].value
        K    = self._build_K(params)
        init = self._init
        return _profiles_from_K_analytic(K, init, t, t0, fwhm)

    def default_params(self, n: int | None = None) -> lmfit.Parameters:
        n = n or self._n
        p = lmfit.Parameters()
        p.add("n_comp", value=n, vary=False)
        p.add("t0", value=0.0, min=-5.0, max=5.0)
        p.add("irf_fwhm", value=0.15, min=0.10, max=0.20, vary=False)
        for (i, j) in self._k_connections:
            tau = 10.0
            p.add(self._tau_key(i, j), value=np.log(tau),
                  min=np.log(0.1), max=np.log(100000.0))
        return p

    def component_labels(self, n: int | None = None) -> list[str]:
        n = n or self._n
        return [f"S{i+1}" for i in range(n)]

    def spectral_type(self) -> str:
        return "SAS"

    def eigenmode_matrix(self, params: lmfit.Parameters):
        try:
            K = self._build_K(params)
            init = (self._init if self._init is not None
                    else np.eye(self._n)[:, 0])
            return _eigenmode_matrix_from_K(K, init)
        except Exception:
            return None
