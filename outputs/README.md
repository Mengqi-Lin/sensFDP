# Generated analysis outputs

Run `notebooks/analyze_simulations.ipynb` or
`analysis/simulation_analysis.py` to regenerate this directory.

## Manuscript-ready outputs from the uploaded files

- `tables/subset_selection_table.tex`: corrected setting order and
  order-invariant averaging over tied maximizing subsets.
- `figures/bound_distributions.pdf`: discrete empirical CDFs for the exact and
  naive FDP bounds.
- `figures/screening_performance.pdf`: screening-call rates by sample size and
  sensitivity parameter.

PNG copies are included for inspection. The corresponding tidy CSV summaries
are in `tables/`, together with `data_audit.csv`.

## Diagnostic output only

`figures/runtime_reduction_legacy_diagnostic.*` and every file whose name
contains `runtime` summarize incomplete legacy results. They must not be used
as publication outputs: the uploaded files contain missing, duplicate, and
truncated records, and the two algorithms' saved rejection fractions disagree.
