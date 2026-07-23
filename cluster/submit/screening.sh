#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

if [[ $# -ne 3 ]]; then
    printf 'Usage: %s GAMMA REPLICATES_PER_TASK NSEEDS\n' "$0" >&2
    exit 2
fi

gamma="$1"
replicates="$2"
number_seeds="$3"

require_number "gamma" "$gamma"
require_positive_integer "replicates_per_task" "$replicates"
require_positive_integer "number_seeds" "$number_seeds"

prepare_run_directories "screening"
gamma_tag="$(safe_number_tag "$gamma")"
submit_array \
    "${SCRIPT_DIR}/../jobs/screening.sbatch" \
    "$number_seeds" \
    "$gamma" "$gamma_tag" "$replicates"
