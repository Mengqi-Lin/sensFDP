#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

if [[ $# -lt 3 ]]; then
    printf 'Usage: %s PAIRS REPLICATES_PER_TASK NSEEDS [SETTING ...]\n' "$0" >&2
    exit 2
fi

pairs="$1"
replicates="$2"
number_seeds="$3"
shift 3
if [[ $# -eq 0 ]]; then
    settings=(6 0 3)
else
    settings=("$@")
fi

require_positive_integer "pairs" "$pairs"
require_positive_integer "replicates_per_task" "$replicates"
require_positive_integer "number_seeds" "$number_seeds"
for setting in "${settings[@]}"; do
    require_nonnegative_integer "setting" "$setting"
    (( 10#$setting <= 8 )) || die "setting must lie between 0 and 8"
done

prepare_run_directories "subset_selection"
settings_tag="$(IFS=-; printf '%s' "${settings[*]}")"
submit_array \
    "${SCRIPT_DIR}/../jobs/subset_selection.sbatch" \
    "$number_seeds" \
    "$pairs" "$replicates" "$settings_tag" "${settings[@]}"
