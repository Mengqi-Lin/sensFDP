#!/usr/bin/env bash
#SBATCH --job-name=fdp-validate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-04:00:00
#SBATCH --mem=8G
#SBATCH --licenses=gurobi@slurmdb:1

set -euo pipefail

[[ $# -eq 5 ]] || { printf 'Internal argument error\n' >&2; exit 2; }
project_root="$1"
source "${project_root}/cluster/job_common.sh"
run_dir="$2"
outcomes="$3"
pairs="$4"
replicates="$5"
seed="$SLURM_ARRAY_TASK_ID"

mkdir -p -- "$run_dir"
load_gurobi

output="${run_dir}/K-${outcomes}_pairs-${pairs}_seed-${seed}.csv"
"$PYTHON_BIN" "${PROJECT_ROOT}/experiments/validate_algorithms.py" \
    --outcomes "$outcomes" \
    --pairs "$pairs" \
    --replicates "$replicates" \
    --gammas 1.25 1.5 1.75 \
    --effect-patterns linear half-null \
    --correlations 0 0.2 \
    --threads "$THREADS" \
    --seed "$seed" \
    --output "$output"
