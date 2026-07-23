#!/usr/bin/env bash

set -euo pipefail
umask 027

CLUSTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${CLUSTER_DIR}/.." && pwd)"

: "${SLURM_ARRAY_TASK_ID:?This job must be submitted as a Slurm array}"

PYTHON_BIN="${PYTHON_BIN:-python}"
THREADS="${SLURM_CPUS_PER_TASK:-1}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

load_gurobi() {
    local gurobi_module="${GUROBI_MODULE-gurobi/10.0.2}"
    if [[ -n "$gurobi_module" ]]; then
        type module >/dev/null 2>&1 \
            || { printf 'Error: the module command is unavailable\n' >&2; exit 2; }
        module load "$gurobi_module"
        export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    fi
}
