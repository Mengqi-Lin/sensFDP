#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${PROJECT_ROOT}/cluster/common.sh"

if [[ $# -ne 2 ]]; then
    printf 'Usage: %s R1_INDICES R2_INDICES\n' "$0" >&2
    printf 'Example: %s 0,1,5,16 1,5,14,16\n' "$0" >&2
    exit 2
fi

prepare_run_directories "application_runtime"
submit_array \
    "${SCRIPT_DIR}/../jobs/runtime.sbatch" \
    14 \
    "$1" "$2"
