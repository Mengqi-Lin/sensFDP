# Slurm jobs

The `submit/` scripts replace the original per-seed submission loops with
Slurm arrays.  Every array task writes its own CSV and metadata JSON, so jobs
never append concurrently to the same file.

## Mapping from the uploaded scripts

| Uploaded name | New launcher | Python entry point |
|---|---|---|
| `compare_vR_naive_vs_exact_expr.sh` | `submit/compare_bounds.sh` | `experiments/compare_bounds.py` |
| `closed_testing_equi.sh` | `submit/validate_algorithms.sh` | `experiments/validate_algorithms.py` |
| `optcall_expr.sh` | `submit/screening.sh` | `experiments/screening.py` |
| `subsets_compete.sh` | `submit/subset_selection.sh` | `experiments/subset_selection.py` |
| `IPcall_KI.sh` | retired | not needed for the current experiments |

The uploaded `settings.py` defines nine configurations, indexed from `0` to
`8`.  The current manuscript uses settings `6`, `0`, and `3`, in that order:
effect-outcome correlations `-0.2`, `0`, and `0.2`, null-outcome correlation
`0.2`, and `Gamma_bias=1.75`.  These indices are encoded explicitly in
`fdp_sensitivity.simulation_settings` and used by the selection job.

## Environment

The jobs use `python` by default and load `gurobi/10.0.2` when optimization is
needed.  A project-specific Python environment is preferable.  Set these
variables before submission when the defaults do not match the cluster:

```bash
export PYTHON_BIN=/path/to/venv/bin/python
export GUROBI_MODULE=gurobi/10.0.2
export MAX_CONCURRENT=4
```

`PYTHONPATH` is set automatically to the package's `src/` directory.  The
screening experiment neither loads Gurobi nor reserves a Gurobi license.

## Submission

Run these commands from any directory:

```bash
# rho, pairs, number of seeds, optional replicates per array task (default 10)
cluster/submit/compare_bounds.sh 0 500 100 10

# outcomes, pairs, replicates per task, number of seeds
# Each task evaluates all 12 manuscript runtime settings.
cluster/submit/validate_algorithms.sh 10 500 10 100

# Gamma, replicates per task, number of seeds
cluster/submit/screening.sh 1.5 10 100

# pairs, replicates per task, number of seeds; defaults to settings 6, 0, 3
cluster/submit/subset_selection.sh 500 10 100

# Run one or more explicit zero-based settings instead.
cluster/submit/subset_selection.sh 500 10 100 6
cluster/submit/subset_selection.sh 500 10 100 0 1 2 3 4 5 6 7 8
```

By default, outputs are placed under `results/raw/<experiment>/<run-id>/` and
logs under `logs/<experiment>/<run-id>/`.  Override `RESULT_ROOT`, `LOG_ROOT`,
or `RUN_ID` before submission if desired.  To inspect an `sbatch` command
without submitting it, set `SBATCH_BIN=echo`.

The runtime job alternates method order, requires exact equality of the two
rejection sets on every dataset, and saves the failing inputs before exiting if
a mismatch occurs.  Its CSV separates the shared screening time from the
method-specific computation time.

Selection summaries can combine every seed file in a run directly:

```bash
python experiments/summarize_selection.py \
    results/raw/subset_selection/<run-id>/*.csv \
    --output results/selection-summary.csv
```

Candidate-subset-level outputs are kept in the run's `candidates/` directory
and are therefore excluded from this glob.

## Resource tuning

The resource headers are conservative starting values based on the uploaded
jobs, not measured optima.  After representative runs, inspect elapsed time and
peak memory with `sacct` and revise `--time` and `--mem`.  Keep the Gurobi
array concurrency cap consistent with the available license pool.
