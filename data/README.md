# data/

This folder is where the application looks for transient-absorption matrices.
To keep the repository code-only, **no experimental data is distributed here.**

- The synthetic demo — **Load demo matrix** — needs no files at all.
- The **Load Raman demo (4 matrices)** button expects these files to be present
  in this folder (it reports cleanly if they are absent):
  - `lyc_THF_AP510RP540.dat`
  - `lyc_THF_AP510RP550.dat`
  - `lyc_THF_AP510RP575.dat`
  - `lyc_THF_AP510RP590.dat`

Any whitespace- or comma-delimited *time × wavelength* matrix
(`.dat` / `.csv` / `.txt`) can be loaded with **Open matrices…**; common
transient-absorption export layouts are auto-detected.
