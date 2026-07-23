#!/usr/bin/env bash
#SBATCH --job-name=fdp-selection
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --licenses=gurobi@slurmdb:1

set -euo pipefail

[[ $# -ge 6 ]] || { printf 'Internal argument error\n' >&2; exit 2; }
project_root="$1"
source "${project_root}/cluster/job_common.sh"
run_dir="$2"
pairs="$3"
replicates="$4"
settings_tag="$5"
shift 5
settings=("$@")
seed="$SLURM_ARRAY_TASK_ID"

mkdir -p -- "$run_dir" "${run_dir}/candidates"
load_gurobi

output="${run_dir}/settings-${settings_tag}_pairs-${pairs}_seed-${seed}.csv"
candidate_output="${run_dir}/candidates/settings-${settings_tag}_pairs-${pairs}_seed-${seed}.csv"
"$PYTHON_BIN" "${PROJECT_ROOT}/experiments/subset_selection.py" \
    --pairs "$pairs" \
    --replicates "$replicates" \
    --settings "${settings[@]}" \
    --precision 0.01 \
    --threads "$THREADS" \
    --seed "$seed" \
    --output "$output" \
    --candidate-output "$candidate_output"
