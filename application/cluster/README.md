# Application Slurm jobs

The application uses one Slurm task per subset and one CSV per task. The
project-level `cluster/common.sh` supplies portable paths, run IDs, log
directories, concurrency limits, and environment settings.

## Submit generalized sensitivity values

```bash
application/cluster/submit/gsv.sh preliminary  # 715 tasks
application/cluster/submit/gsv.sh full         # 5,985 tasks
```

Set `MAX_CONCURRENT` to the number of Gurobi licenses/jobs you may use. Each
task requests one license and one CPU by default. Adjust the `#SBATCH` resource
headers after inspecting representative jobs with `sacct`.

## Submit the runtime table

After the corrected subsets have been selected:

```bash
application/cluster/submit/runtime.sh 0,1,5,16 1,5,14,16
```

The indices above are only the legacy examples. Use the corrected indices from
`application/outputs/headline_subsets.csv`. There are 14 tasks: two subsets by
seven Gamma values. Execution order alternates across tasks, and every task
requires exact equality of the two rejection sets.

## Dry run

Set `SBATCH_BIN=echo` to inspect submission commands without submitting:

```bash
SBATCH_BIN=echo application/cluster/submit/gsv.sh preliminary
```

Results and logs are placed in timestamped folders under `results/raw/` and
`logs/`. Aggregation is a separate step and fails on missing or duplicate jobs.
