# Provenance and references

This program performs global and target analysis of transient-absorption data
by variable projection. The numerical formalism follows established, published
methods; the implementation is original (NumPy / SciPy) and MIT-licensed.

- **Variable projection** (separation of the linear associated-spectra
  parameters from the nonlinear kinetic parameters):
  Golub & Pereyra (1973), *SIAM J. Numer. Anal.* **10**, 413–432.

- **Compartmental target analysis, the column-based rate matrix K, and the
  analytical exp–Gaussian (erfc) IRF-convolution formulas:**
  van Stokkum, Larsen & van Grondelle (2004), *BBA Bioenergetics* **1657**,
  82–104, doi:10.1016/j.bbabio.2004.04.011.

- **Bundled Raman demo data** (the four lycopene-in-THF FSRRS matrices in
  `data/`):
  Bercy, R.; D'mello, V. C.; Gall, A.; Ilioaia, C.; Pascal, A. A.; Romero,
  J. J.; Robert, B.; Llansola-Portoles, M. J., Reassessing Carotenoid
  Photophysics: Shedding Light on Dark States. J. Am. Chem. Soc. 2026, 148
  (23), 23976-23985.

No source code from Glotaran, pyglotaran or TIMP was reused.

Lifetimes are optimised in log-space (`log_tau = ln(τ/ps)`) so the Jacobian
stays well-conditioned when lifetimes span several orders of magnitude.
