#!/usr/bin/env bash
#
# Measure the NaN rate of the A1 rough-terrain reproducer in the current worktree.
#
# The failure is stochastic (Newton runs with deterministic_mode="not_guaranteed"), so a
# single run says little. This runs N independent 60-iteration attempts and reports how
# many aborted on the rsl_rl NaN guard, plus the iteration each one died at.
#
#   ./rate.sh <label> [n]
#
set -uo pipefail
LABEL=${1:?usage: rate.sh <label> [n]}
N=${2:-10}
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

mkdir -p rate_logs
fails=0
iters=()
for i in $(seq 1 "${N}"); do
    seed=$((41 + i))
    log="rate_logs/${LABEL}_s${seed}.log"
    timeout 900 uv run --no-sync isaaclab train --rl_library rsl_rl \
        --task IsaacContrib-Velocity-Rough-UnitreeA1 --seed "${seed}" \
        --max_iterations 60 --run_name "rate_${LABEL}_s${seed}" presets=newton_mjwarp >"${log}" 2>&1
    if grep -q "contains NaN values" "${log}"; then
        fails=$((fails + 1))
        it=$(sed 's/\x1b\[[0-9;]*m//g' "${log}" | grep -oE "Learning iteration [0-9]+/" | tail -1 | tr -d 'A-Za-z /')
        iters+=("${it:-0}")
        printf '  seed %-3s NaN @ iter %s\n' "${seed}" "${it:-0}"
    else
        printf '  seed %-3s clean\n' "${seed}"
    fi
done

echo "RESULT ${LABEL}: ${fails}/${N} NaN   died_at=[${iters[*]:-}]"
