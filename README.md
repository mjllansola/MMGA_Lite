# Global fitting — transient-absorption global & target analysis

> **Reference:** Bercy, R.; D'mello, V. C.; Gall, A.; Ilioaia, C.; Pascal,
> A. A.; Romero, J. J.; Robert, B.; Llansola-Portoles, M. J., Reassessing
> Carotenoid Photophysics: Shedding Light on Dark States. *J. Am. Chem. Soc.*
> **2026**, *148* (23), 23976-23985.
>
> **Preprint:** arXiv link TBA

A minimal desktop application for **global and target analysis** of
transient-absorption (TA) data. Load one or more time × wavelength matrices,
explore them, and fit a kinetic model whose lifetimes are shared across all
loaded matrices. The linear associated spectra (DAS / SAS / EADS) are recovered
by variable projection.

The scope is intentionally small: this is the modelling core, not a data
pipeline. There is no pre-processing, no publication layout engine.

## Features

- **Load matrices** — whitespace- or comma-delimited text files (`.dat`,
  `.csv`, `.txt`). Common TA export layouts are auto-detected.
- **Explore** — a carpet plot with a spectrum-at-time slice and a
  kinetics-at-wavelength slice, driven by two slide bars. View only; there are
  no processing controls. Kinetics use a split time axis: linear from −3 ps to
  1 ps, then logarithmic — the usual ultrafast-TA convention.
- **Manage matrices** — remove one or several loaded matrices (Ctrl/Shift-click
  to multi-select, then **Remove matrix**).
- **Remove chirp** — a manual group-velocity-dispersion correction. In the
  popup, left-click the carpet to place time-zero points (a fine-picking trace
  popup helps set each t0 exactly), right-click to remove the nearest; a
  polynomial is fitted through your points and every wavelength is shifted to a
  common t0. Points are placed **manually only — there is no auto-detection.**
- **Fit** three kinetic models, with the kinetics shared globally across every
  loaded matrix:
  - **Parallel** (sum of independent exponentials) → DAS
  - **Sequential** (A→B→C→…) → EADS
  - **Target** (arbitrary compartmental scheme) → SAS
- **Inspect results** in four tabs — associated spectra (DAS/SAS/EADS), fitted
  traces (data vs. fit), concentration profiles, and residuals.
- **Export everything** — every plot exports to PNG/SVG/PDF (matplotlib
  toolbar) and its underlying numbers to CSV.

## Install

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py                 # start empty
python main.py path/to/file.dat   # open a matrix on startup
```

Then click **Load demo matrix** for a synthetic 3-component dataset (needs no
files), move the slide bars to explore, and click **Global fit…** → **Run
fit**. This is the quickest way to see the whole workflow end to end.

To load your own data, use **Open matrices…** (whitespace/comma-delimited
`.dat`/`.csv`/`.txt`; common TA export layouts are auto-detected). When several
matrices are loaded, their kinetics are fit **simultaneously**.

### Data is not bundled

To keep the repository code-only, no experimental matrices are distributed
here. Two demo loaders look for files in `data/` and simply report if they are
absent (see [`data/README.md`](data/README.md) for the expected filenames):

- **Load demo matrix** — synthetic, always available (no file needed).
- **Load Raman demo (4 matrices)** — expects four lycopene FSRRS matrices in
  `data/`; drop them in to exercise the multi-matrix global fit. The fit dialog
  opens preconfigured for this case (a 4-species sequential scheme with
  starting lifetimes and per-lifetime limits).

## Fitting workflow

1. Open one or more matrices. When several are loaded, their kinetic
   parameters are fit **simultaneously** (shared lifetimes / rates / t0 / IRF);
   each matrix keeps its own associated spectra.
2. In **Global fit…**, pick a model and the number of components. It opens on
   a 4-species **sequential** scheme by default.
   - For **Target**, describe the scheme in the connections box, e.g.
     `1>2, 2>3, 3>G`. `a>b` is transfer from compartment *a* to *b*; `a>G` is
     decay of *a* to the ground state.
3. Edit the initial lifetimes and their **Min/Max** limits in the parameter
   table, and toggle which parameters vary. Optionally constrain the spectra to
   be non-negative (NNLS).
4. **Run fit.** Progress and the recovered lifetimes appear in the status line;
   the result tabs populate. Use the **Matrix** selector to switch between
   loaded matrices and the **View** selector to switch DAS/SAS/EADS.

## Method

For fixed nonlinear parameters θ (lifetimes, rates, t0, IRF width), the
concentration profiles C(t; θ) are computed analytically (exp–Gaussian IRF
convolution), and the associated spectra are the least-squares (or
non-negative least-squares) solution of `Data ≈ C · S` per wavelength. The
residual is handed to a Levenberg–Marquardt / trust-region optimiser (`lmfit`
+ SciPy `least_squares`), which refines θ. Lifetimes are optimised in
log-space for numerical conditioning. See `NOTICE.md` for references.

## Layout

```
main.py            entry point
core/
  data.py          TADataset: file loading and slicing
  models.py        Parallel / Sequential / Target kinetic models + IRF
  fitting.py       GlobalFitter (variable projection), FitResult
  dispersion.py    manual chirp correction (polynomial fit + column shift)
gui/
  main_window.py   matrix loader + viewer
  plot_widgets.py  exportable figure, matrix viewer with slide bars
  fit_dialog.py    model setup, threaded fit, result tabs
  chirp_dialog.py  manual chirp-correction popup (no auto-guess)
data/              sample matrices
```

## License

MIT — see `LICENSE`.
