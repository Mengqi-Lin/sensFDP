#!/usr/bin/env bash
#SBATCH --job-name=fdp-app-gsv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --mem=8G
#SBATCH --licenses=gurobi@slurmdb:1

set -euo pipefail

[[ $# -eq 5 ]] || { printf 'Internal argument error\n' >&2; exit 2; }
project_root="$1"
source "${project_root}/cluster/job_common.sh"
run_dir="$2"
scope="$3"
family_size="$4"
subset_size="$5"

mkdir -p -- "$run_dir"
load_gurobi

output="${run_dir}/${scope}_subset-${SLURM_ARRAY_TASK_ID}.csv"
"$PYTHON_BIN" "${project_root}/application/scripts/run_application_gsv.py" \
    --scores "${project_root}/application/data/derived/application_scores.csv" \
    --nominal-pvalues "${project_root}/application/data/derived/nominal_pvalues.csv" \
    --scope "$scope" \
    --subset-index "$SLURM_ARRAY_TASK_ID" \
    --subset-size "$subset_size" \
    --precision 0.01 \
    --threads "$THREADS" \
    --output "$output"
