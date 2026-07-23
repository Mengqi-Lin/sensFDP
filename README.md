# sensFDP

This is the unified research bundle for simultaneous FDP sensitivity analysis.
It contains the reusable implementation, simulation studies, cluster jobs,
saved simulation results, manuscript-output notebooks, and the cleaned WLS
application workflow. Snapshots of the largest legacy modules are retained in
`legacy/` for comparison.

The outer project is named `sensFDP`. The installable Python distribution and
import package remain `fdp-sensitivity` and `fdp_sensitivity`, respectively, so
existing scripts do not need to change.

## Layout

- `src/fdp_sensitivity/`: reusable statistical and optimization routines.
- `experiments/`: one entry point per manuscript experiment.
- `cluster/`: Slurm array launchers and job scripts for the experiments.
- `data/`: untouched copies of the uploaded legacy simulation outputs.
- `analysis/`: validated parsers and manuscript-output functions.
- `notebooks/analyze_simulations.ipynb`: guided, reproducible data analysis.
- `application/`: cleaned WLS application workflow, pinned inputs, corrected
  score construction, Slurm jobs, and a manuscript-analysis notebook. This is
  part of the main bundle and does not require a separate installation.
- `outputs/`: generated tables and figures.
- `tests/`: small deterministic checks, including equivalence checks between
  the enumerative and mixed-integer implementations.
- `CODE_AUDIT.md`: mapping from the uploaded files to the manuscript and a list
  of issues found during review.

The intended installation is

```bash
python -m pip install -e .
```

For the notebook and figures, install the analysis dependencies with

```bash
python -m pip install -e '.[analysis]'
```

Gurobi and a working Gurobi license are required for the optimization routines.
The current review environment does not contain `gurobipy`, so Gurobi-dependent
tests must be run in the authors' licensed environment.

## Reproducibility principles

1. Each Monte Carlo replicate is identified by an explicit seed.
2. Results are written in tidy form: one row per replicate and simulation
   setting, rather than Python lists embedded inside CSV cells.
3. The enumerative and mixed-integer procedures are compared on every paired
   dataset.  A mismatch raises an error and saves the failing replicate.
4. Solver status, runtime, and the number of optimization calls are retained.
5. The full-matching and matched-pair paths are distinguished explicitly.

See `cluster/README.md` for cluster setup, submission commands, output paths,
and the mapping from the original job names.

The application has its own entry point and audit at
`application/README.md`. Its legacy numerical results are quarantined because
the uploaded notebook constructed the spouse score from alcohol and the old
full-matching screening routine was not exact. The corrected source data and
score matrix are included; application sensitivity values require a licensed
cluster rerun.

## Analyze the uploaded results

Open `notebooks/analyze_simulations.ipynb`, or regenerate all outputs from the
command line:

```bash
PYTHONPATH=src MPLBACKEND=Agg python analysis/simulation_analysis.py
```

The notebook starts with a file-level audit. It therefore keeps incomplete or
internally inconsistent legacy results visible instead of silently dropping or
repairing them.
