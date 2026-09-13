#!/usr/bin/env bash
#
# Keep re-running the armed capture until one attempt actually dies on the NaN.
#
# A 512-environment run reproduces the abort roughly half the time within 60 iterations,
# so a single attempt is not enough. Each attempt writes its own clip; the loop stops at
# the first attempt that aborts (exit 1) with frames buffered, which is the one whose clip
# holds the blow-up rather than a survived contact spike.
#
set -uo pipefail
cd /home/henry/workspace/IsaacLab-develop || exit 1

export TRIGGER_N=${TRIGGER_N:-20000}
export ARM_STEPS=${ARM_STEPS:-6}
export CLIP_FPS=${CLIP_FPS:-5}
export VIZ_WIDTH=${VIZ_WIDTH:-960}
export VIZ_HEIGHT=${VIZ_HEIGHT:-540}

for seed in 42 43 44 45 46 47 48 49; do
    out="nanvid/nanshot_s${seed}.mp4"
    log="nanvid/hunt_s${seed}.log"
    echo "[HUNT] seed ${seed} -> ${log}"
    OUT_MP4="${out}" uv run --no-sync python nanvid/nanshot.py \
        --rl_library rsl_rl --task IsaacContrib-Velocity-Rough-UnitreeA1 \
        --num_envs 512 --seed "${seed}" --max_iterations 60 \
        --run_name "hunt_s${seed}" --viz newton presets=newton_mjwarp >"${log}" 2>&1
    rc=$?
    nan=$(grep -c "contains NaN values" "${log}")
    wrote=$(grep -c "NANSHOT: wrote" "${log}")
    echo "[HUNT] seed ${seed}: exit=${rc} nan=${nan} clip_written=${wrote}"
    if [[ ${nan} -gt 0 && ${wrote} -gt 0 ]]; then
        echo "[HUNT] captured the blow-up on seed ${seed}: ${out}"
        grep -E "NANSHOT: (armed #|non-finite|wrote)" "${log}" | tail -4
        exit 0
    fi
done

echo "[HUNT] no attempt aborted with frames buffered"
exit 1
