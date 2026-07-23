#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

if [[ $# -ne 4 ]]; then
    printf 'Usage: %s OUTCOMES PAIRS REPLICATES_PER_TASK NSEEDS\n' "$0" >&2
    exit 2
fi

outcomes="$1"
pairs="$2"
replicates="$3"
number_seeds="$4"

require_positive_integer "outcomes" "$outcomes"
require_positive_integer "pairs" "$pairs"
require_positive_integer "replicates_per_task" "$replicates"
require_positive_integer "number_seeds" "$number_seeds"

prepare_run_directories "validate_algorithms"
submit_array \
    "${SCRIPT_DIR}/../jobs/validate_algorithms.sbatch" \
    "$number_seeds" \
    "$outcomes" "$pairs" "$replicates"
