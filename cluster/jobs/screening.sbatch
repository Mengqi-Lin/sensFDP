#!/usr/bin/env bash
#SBATCH --job-name=fdp-screening
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4G

set -euo pipefail

[[ $# -eq 5 ]] || { printf 'Internal argument error\n' >&2; exit 2; }
project_root="$1"
source "${project_root}/cluster/job_common.sh"
run_dir="$2"
gamma="$3"
gamma_tag="$4"
replicates="$5"
seed="$SLURM_ARRAY_TASK_ID"

mkdir -p -- "$run_dir"
output="${run_dir}/gamma-${gamma_tag}_seed-${seed}.csv"

"$PYTHON_BIN" "${PROJECT_ROOT}/experiments/screening.py" \
    --pairs 500 1000 2000 5000 10000 \
    --gammas "$gamma" \
    --outcomes 10 \
    --replicates "$replicates" \
    --seed "$seed" \
    --output "$output"
