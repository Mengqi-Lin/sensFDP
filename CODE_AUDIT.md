# Code audit

## Manuscript-to-code map

| Manuscript component | Uploaded implementation |
|---|---|
| Score construction | `PO_to_Qmat.py` |
| Paired-data generator | `generate_data.py` |
| Core MIQCP and screening | `sensitivityMCP.py` |
| Exact-versus-naive FDP-bound simulation | `compare_vR_naive_vs_exact_expr.py` calling `expr_function.compare_vR_naive_vs_exact_expr` |
| Enumerative-versus-MIQCP runtime simulation | `closed_testing_equi.py` calling `expr_function.closed_testing_equi` |
| Screening simulation | `optcall_expr.py` calling `expr_function.optcall_expr` |
| Subset-selection simulation | `expr_function.subsets_compete` with the configurations in `settings.py` |
| Alternative worst-p-value solver | `solve_worst_pval.py` |
| Older naive sensitivity-value routine | `gSensitivity_value_naive.py` |

## Material findings

1. `solve_vR` uses `any(worst_pvalues > alpha)` when constructing the lower
   bound for a requested subset.  The check must be restricted to outcomes in
   that subset.  Otherwise an outcome outside the reported subset can force the
   returned FDP bound to be at least one.
2. The active analytic `worst_pval` routine is appropriate for the paired
   simulations, but it is not generally the exact maximizer of the normally
   approximated two-sided p-value for matched sets of size greater than two.
   Exact full-matching screening should use the convex minimum-zeta oracle (or a
   bisection based on it).
3. `solve_worst_pval.py` uses marginal lower and upper bounds on each treatment
   probability.  For sets of size greater than two these bounds do not enforce
   every pairwise ratio constraint.  For example, with `Gamma=2`, the vector
   `(0.2, 0.3, 0.5)` passes the box constraints but has ratio `2.5`.
4. The runtime experiment records average rejection proportions but does not
   require equality of the two rejection sets on each paired dataset.  This can
   conceal algorithmic or numerical mismatches.
5. Several old experiment functions are not executable as written:
   `solve_vR_singleton_equi` and `pseudo_vR_sensitivity_expr` contain undefined
   names, while `gSensitivity_value_naive.py` has missing imports and calls an
   obsolete `data_process` signature.
6. The uploaded cluster launchers submit one job per seed and several Python
   drivers append concurrently to a shared CSV.  This risks corrupted or
   duplicated results.  They also depend on the submission directory, pass
   unquoted values through `--export`, and do not constrain solver or BLAS
   threads.
7. `IPcall_KI` is not needed for the current manuscript experiments and has
   been retired.  Although the old `subsets_compete.py` driver was not
   uploaded, `settings.py` identifies all nine configurations.  The current
   manuscript uses zero-based setting indices `6`, `0`, and `3`.

## Safe implementation fixes

- Convert score arrays to floating point before assigning M-scores.
- Relabel arbitrary matched-set identifiers internally instead of assuming
  labels are `0, ..., B-1`.
- Reject zero-variance or non-finite score columns with an informative error.
- Restrict the lower bound for `v_R` to the requested subset.
- Write tidy simulation output and preserve solver status and paired equality
  diagnostics.
- Submit seeds as a Slurm array, write one file per task, resolve project paths
  explicitly, and avoid reserving a Gurobi license for analytic screening.

## Checks still requiring the licensed environment

- Verify exact equality of enumerative and MIQCP rejection sets for every
  replicate used in the runtime figure.
- Recompute full-matching application results using exact two-sided
  worst-p-value screening.
- Record Gurobi version, thread count, tolerances, hardware, and any time limit.
- Decide and document the numerical convention for the strict condition
  `min zeta < 0`.

## WLS application audit

The cleaned application workflow is in `application/`. The uploaded notebook
used the alcohol outcome twice when constructing the columns labeled alcohol
and spouse. The same 13 outcomes remain nominally significant after correcting
spouse, but all joint sensitivity values, subset rankings, and runtimes must be
recomputed. The old full-matching worst-p-value solver also failed to impose all
pairwise Rosenbaum restrictions, and the stored full-family size-four table is
missing nine of 5,985 subsets. Historical outputs are retained only under
`application/legacy_unvalidated/`.

The exact R matching script was not uploaded. Although `matched_index.csv` is
structurally valid, five covariates contain negative WLS sentinel codes and the
folder does not show how these produced the four missingness indicators stated
in the manuscript. Recovering the matching and balance code remains necessary.

## Uploaded simulation-output audit

The legacy CSVs have been copied unchanged to `data/` and are parsed by
`analysis/simulation_analysis.py`.

- All nine subset-selection files are complete. The manuscript settings are
  files `6`, `0`, and `3` for effect-outcome correlations `-0.2`, `0`, and
  `0.2`; the current manuscript table appears to attach these labels to the
  wrong files. Because sensitivity values were saved on a `0.01` grid, tied
  maximizing subsets are common. The analysis reports the order-invariant
  expected success under uniform tie-breaking.
- `Gamma_1.25.csv` is missing screening seed `60` (990 rather than 1,000
  datasets), and the exact-versus-naive file for `rho=0.2` is missing seed `54`
  (990 datasets). The remaining files in those experiments are complete. For
  `Gamma=1.5`, both screening rates are nonmonotone over the saved sample-size
  grid, so these outputs do not support a blanket finite-sample claim that the
  rates decrease with sample size.
- The `K=10` runtime file has 955 records, one conflicting duplicate seed, and
  an unexpected seed `1000`. The `K=20` file has only 606 complete records and
  ends in a truncated row. Moreover, the stored rejection fractions disagree
  between algorithms in 2.7% and 14.5% of setting-record cells, respectively.
  Since the rows were appended in job-completion order, the partial-file means
  may also be completion-time biased. The rejection grids strongly suggest one
  simulation per stored record, but `nsims` was not saved. Consequently,
  runtime summaries and their plot are marked diagnostic only; they should not
  replace the manuscript runtime result without a validated rerun.
