#!/usr/bin/env bash
#
# Measure the A1 rough-terrain NaN rate under a terrain/margin variant.
#
# Same protocol as rate.sh, but routed through armprobe.py so the collision
# representation (ARM) and the Newton shape margin (MARGIN) can be pinned, and the
# realized terrain is asserted rather than assumed.
#
#   ARM=mesh ./rate_arm.sh mesh 12
#   MARGIN=0.01 ./rate_arm.sh hf_margin001 12
#
set -uo pipefail
LABEL=${1:?usage: rate_arm.sh <label> [n]}
N=${2:-12}
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

mkdir -p rate_logs
fails=0
iters=()
for i in $(seq 1 "${N}"); do
    seed=$((41 + i))
    log="rate_logs/${LABEL}_s${seed}.log"
    timeout 900 uv run --no-sync python g1_hf_check/armprobe.py --rl_library rsl_rl \
        --task IsaacContrib-Velocity-Rough-UnitreeA1 --seed "${seed}" \
        --max_iterations 60 --run_name "rate_${LABEL}_s${seed}" presets=newton_mjwarp >"${log}" 2>&1
    # A run that never confirmed its arm is not a valid sample.
    if ! grep -q "ARMPROBE: CONFIRMED" "${log}"; then
        printf '  seed %-3s INVALID (arm not confirmed)\n' "${seed}"
        continue
    fi
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
