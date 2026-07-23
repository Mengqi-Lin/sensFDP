#!/usr/bin/env bash

set -euo pipefail

CLUSTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${CLUSTER_DIR}/.." && pwd)"

SBATCH_BIN="${SBATCH_BIN:-sbatch}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/raw}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer"
}

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "${name} must be a nonnegative integer"
}

require_number() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] \
        || die "${name} must be numeric"
}

safe_number_tag() {
    local value="$1"
    value="${value//-/m}"
    value="${value//+/p}"
    value="${value//./p}"
    printf '%s' "$value"
}

prepare_run_directories() {
    local experiment="$1"
    RUN_DIR="${RESULT_ROOT}/${experiment}/${RUN_ID}"
    LOG_DIR="${LOG_ROOT}/${experiment}/${RUN_ID}"
    mkdir -p -- "$RUN_DIR" "$LOG_DIR"
}

submit_array() {
    local job_script="$1"
    local number_seeds="$2"
    shift 2

    require_positive_integer "number_seeds" "$number_seeds"
    require_positive_integer "MAX_CONCURRENT" "$MAX_CONCURRENT"
    [[ -f "$job_script" ]] || die "job script not found: ${job_script}"

    "$SBATCH_BIN" \
        --parsable \
        --array="0-$((number_seeds - 1))%${MAX_CONCURRENT}" \
        --chdir="$PROJECT_ROOT" \
        --output="${LOG_DIR}/%x-%A_%a.out" \
        --error="${LOG_DIR}/%x-%A_%a.err" \
        "$job_script" "$PROJECT_ROOT" "$RUN_DIR" "$@"

    printf 'Results: %s\nLogs: %s\n' "$RUN_DIR" "$LOG_DIR"
}
