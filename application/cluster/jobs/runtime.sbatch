#!/usr/bin/env bash
#SBATCH --job-name=fdp-app-runtime
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --mem=8G
#SBATCH --licenses=gurobi@slurmdb:1

set -euo pipefail

[[ $# -eq 4 ]] || { printf 'Internal argument error\n' >&2; exit 2; }
project_root="$1"
source "${project_root}/cluster/job_common.sh"
run_dir="$2"
r1="$3"
r2="$4"

gammas=(1.1 1.2 1.3 1.4 1.5 1.6 1.7)
subset_position=$((SLURM_ARRAY_TASK_ID / 7))
gamma_position=$((SLURM_ARRAY_TASK_ID % 7))
if (( subset_position == 0 )); then
    subset="$r1"
    label="R1"
else
    subset="$r2"
    label="R2"
fi
gamma="${gammas[$gamma_position]}"
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
    order="miqcp-first"
else
    order="enumerative-first"
fi

mkdir -p -- "$run_dir"
load_gurobi
output="${run_dir}/${label}_gamma-${gamma}.csv"
"$PYTHON_BIN" "${project_root}/application/scripts/run_application_runtime.py" \
    --scores "${project_root}/application/data/derived/application_scores.csv" \
    --subset "$subset" \
    --subset-label "$label" \
    --gamma "$gamma" \
    --order "$order" \
    --threads "$THREADS" \
    --output "$output"
