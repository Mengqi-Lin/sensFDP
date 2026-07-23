#!/usr/bin/env bash
#SBATCH --job-name=fdp-bounds
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --licenses=gurobi@slurmdb:1

set -euo pipefail

[[ $# -eq 6 ]] || { printf 'Internal argument error\n' >&2; exit 2; }
project_root="$1"
source "${project_root}/cluster/job_common.sh"
run_dir="$2"
rho="$3"
rho_tag="$4"
pairs="$5"
replicates="$6"
seed="$SLURM_ARRAY_TASK_ID"

mkdir -p -- "$run_dir"
load_gurobi

output="${run_dir}/rho-${rho_tag}_pairs-${pairs}_seed-${seed}.csv"
"$PYTHON_BIN" "${PROJECT_ROOT}/experiments/compare_bounds.py" \
    --rho "$rho" \
    --pairs "$pairs" \
    --replicates "$replicates" \
    --gammas 1 1.25 1.5 1.75 \
    --threads "$THREADS" \
    --seed "$seed" \
    --output "$output"
