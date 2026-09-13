#!/usr/bin/env bash
#
# Reproduction attempt for the OSMO "Newton MJWarp: NaN divergence on rough-terrain
# locomotion" report, on plain origin/develop instead of antoiner/feat/odin-v2.
#
# Runs the three configurations the report names, back to back, with stock task
# defaults and the same seeds. rsl_rl's `check_for_nan` guard defaults to True, so a
# NaN in obs/reward/done aborts the run with a non-zero exit code and a traceback.
#
#   ./g1_hf_check/repro.sh
#
set -uo pipefail

cd /home/henry/workspace/IsaacLab-develop || exit 1
OUT=g1_hf_check/repro
mkdir -p "${OUT}"

run() {
    local task=$1 seed=$2 name=$3
    local log="${OUT}/${name}.log"

    echo "[RUN ] ${name}  task=${task} seed=${seed}  $(date -Is)"
    {
        echo "# task=${task} seed=${seed} preset=newton_mjwarp"
        echo "# commit=$(git rev-parse HEAD)"
        echo "# started=$(date -Is)"
    } >"${OUT}/${name}.meta"

    local t0
    t0=$(date +%s)
    uv run isaaclab train --rl_library rsl_rl --task "${task}" --seed "${seed}" \
        --run_name "${name}" presets=newton_mjwarp >"${log}" 2>&1
    local rc=$?
    local elapsed=$(($(date +%s) - t0))

    # The report's signature is an rsl_rl ValueError on NaN observations.
    local nan_hit="no"
    grep -q "contains NaN values" "${log}" && nan_hit="**YES**"
    local iters
    iters=$(sed 's/\x1b\[[0-9;]*m//g' "${log}" | grep -oE 'Learning iteration [0-9]+/[0-9]+' | tail -n 1)

    {
        echo "exit=${rc}"
        echo "elapsed=${elapsed}s"
        echo "nan_abort=${nan_hit}"
        echo "last=${iters:-<none>}"
    } >"${OUT}/${name}.status"

    echo "[DONE] ${name} rc=${rc} nan=${nan_hit} ${elapsed}s (${iters:-no iterations})"
    echo
}

run IsaacContrib-Velocity-Rough-UnitreeA1 42 a1_rough_s42
run Isaac-Velocity-Rough-G1               43 g1_rough_s43
run Isaac-Velocity-Rough-G1               44 g1_rough_s44

echo "=== summary ==="
for f in "${OUT}"/*.status; do
    echo "--- $(basename "${f}" .status)"
    cat "${f}"
done
