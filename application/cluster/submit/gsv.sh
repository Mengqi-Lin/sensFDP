#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${PROJECT_ROOT}/cluster/common.sh"

if [[ $# -ne 1 || ( "$1" != "preliminary" && "$1" != "full" ) ]]; then
    printf 'Usage: %s preliminary|full\n' "$0" >&2
    exit 2
fi

scope="$1"
subset_size=4
if [[ "$scope" == "preliminary" ]]; then
    family_size=13
    task_count=715
else
    family_size=21
    task_count=5985
fi

prepare_run_directories "application_${scope}_gsv"
submit_array \
    "${SCRIPT_DIR}/../jobs/gsv.sbatch" \
    "$task_count" \
    "$scope" "$family_size" "$subset_size"
