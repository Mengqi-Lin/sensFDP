#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

if [[ $# -lt 3 || $# -gt 4 ]]; then
    printf 'Usage: %s RHO PAIRS NSEEDS [REPLICATES_PER_TASK]\n' "$0" >&2
    exit 2
fi

rho="$1"
pairs="$2"
number_seeds="$3"
replicates="${4:-10}"

require_number "rho" "$rho"
require_positive_integer "pairs" "$pairs"
require_positive_integer "number_seeds" "$number_seeds"
require_positive_integer "replicates_per_task" "$replicates"

prepare_run_directories "compare_bounds"
rho_tag="$(safe_number_tag "$rho")"
submit_array \
    "${SCRIPT_DIR}/../jobs/compare_bounds.sbatch" \
    "$number_seeds" \
    "$rho" "$rho_tag" "$pairs" "$replicates"
