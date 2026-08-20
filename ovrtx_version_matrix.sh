#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Benchmark the OVRTX renderer across two ovrtx runtimes:
#
#   ovrtx_0_4_1  the version pinned in pyproject.toml (public wheel)
#   ovrtx_0_5_0  the internal wheel under ~/dev/wheels/ov
#
# Each runtime is measured once per camera preset in CAMERA_PRESETS, so the matrix is
# (ovrtx version) x (render mode) x REPEATS. Both runtimes run against whatever ovstage the
# project has pinned; the script never touches it.
#
# The ovrtx wheel is swapped in-place in ./.venv between configurations, so this MUTATES the
# working environment. The baseline pin is restored on exit, including on failure or Ctrl-C.
#
# On a fresh clone -- or after a ``git clean -xfd`` has removed .venv -- the environment is
# created with ``uv sync --extra ov`` before anything else runs. Set BOOTSTRAP=0 to make a
# missing environment a hard error instead.
#
# Usage:
#   ./ovrtx_version_matrix.sh                          # both runtimes, both render modes
#   ./ovrtx_version_matrix.sh ovrtx_0_5_0              # a subset of runtimes, in order
#   ENVS=512 STEPS=300 ./ovrtx_version_matrix.sh       # override the workload
#   CAMERA_PRESETS="rgb128" ./ovrtx_version_matrix.sh  # a single render mode
#   TASK_NAME=Isaac-Cartpole-Camera-Direct BASE_PRESETS=newton_mjwarp,ovrtx \
#       CAMERA_PRESETS="rgb simple_shading_full_mdl" ./ovrtx_version_matrix.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------
TASK_NAME="${TASK_NAME:-Isaac-Lift-KukaAllegro-Camera}"
# The preset tokens held fixed across the matrix. The camera preset is appended per render mode.
BASE_PRESETS="${BASE_PRESETS:-newton_mjwarp,ovrtx_renderer,single_camera}"
# Render modes to sweep, space separated. These are camera preset tokens, so the spelling is
# task-specific: Kuka Allegro carries a resolution suffix (rgb128), Cartpole and Shadow Hand do
# not (rgb).
CAMERA_PRESETS="${CAMERA_PRESETS:-rgb128 simple_shading_full_mdl128}"
ENVS="${ENVS:-1024}"
STEPS="${STEPS:-150}"
WARMUP="${WARMUP:-50}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-data}"
# Repeats per configuration. A single run cannot separate a real regression from run-to-run noise,
# which on this workload is a few percent of the mean iteration time.
REPEATS="${REPEATS:-3}"

read -r -a MODES <<<"$CAMERA_PRESETS"
[[ ${#MODES[@]} -gt 0 ]] || { echo "CAMERA_PRESETS is empty" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Runtimes
# ---------------------------------------------------------------------------
VENV_PY="${VENV_PY:-$REPO_ROOT/.venv/bin/python}"
OVRTX_050_WHEEL="${OVRTX_050_WHEEL:-$HOME/dev/wheels/ov/ovrtx-0.5.0.0-py3-none-manylinux_2_35_x86_64.whl}"
# ovrtx resolves from the pypi-public index (see [tool.uv.sources] in pyproject.toml).
OVRTX_INDEX_URL="${OVRTX_INDEX_URL:-https://pypi.org/simple}"
# Extras to sync when .venv is missing. ``ov`` is the minimum this benchmark needs: newton arrives
# with the base workspace dependencies (isaaclab-newton), and ``ov`` adds ovrtx/ovstage/ovphysx.
# ``all`` also works but drags in Isaac Sim, which this Newton + OVRTX workload never touches.
UV_SYNC_EXTRAS="${UV_SYNC_EXTRAS:-ov}"
# Set to 0 to refuse to create or repair .venv and fail with instructions instead.
BOOTSTRAP="${BOOTSTRAP:-1}"

log() { printf '\n[ovrtx-matrix] %s\n' "$*"; }
die() { printf '\n[ovrtx-matrix] ERROR: %s\n' "$*" >&2; exit 1; }

installed_version() {
    "$VENV_PY" -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || echo "none"
}

# Baseline pin, read from the single source of truth so it cannot drift from the lockfile.
pinned_version() {
    "$VENV_PY" - "$1" <<'PY'
import pathlib
import sys
import tomllib

data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
print(data["tool"]["isaaclab"]["versions"][sys.argv[1]])
PY
}

# ``uv pip install`` targets .venv directly. The benchmark is likewise invoked through
# ``$VENV_PY`` rather than ``uv run``, which would re-sync the environment and undo the swap.
install_ovrtx_spec() {
    local spec="$1"
    log "installing $spec"
    # --no-deps keeps the swap surgical. The 0.5.0 wheel happens to declare no dependencies today,
    # but the published ovrtx does, and resolving them would let ovstage move underneath a matrix
    # that exists to vary ovrtx alone.
    uv pip install --python "$VENV_PY" --index-url "$OVRTX_INDEX_URL" --no-deps "$spec"
}

ensure_ovrtx() {
    local want_version="$1" spec="$2"
    if [[ "$(installed_version ovrtx)" == "$want_version" ]]; then
        log "ovrtx $want_version already installed"
        return
    fi
    install_ovrtx_spec "$spec"
    local now
    now="$(installed_version ovrtx)"
    [[ "$now" == "$want_version" ]] || die "expected ovrtx $want_version after install, got $now"
}

# Create or repair .venv so the script works on a fresh clone, where ``git clean -xfd`` or a
# first checkout leaves no environment at all. Everything downstream reads pyproject.toml and the
# installed distributions through $VENV_PY, so this has to succeed before any of it runs.
bootstrap_env() {
    command -v uv >/dev/null 2>&1 || die \
        "uv is not on PATH. Install it with 'curl -LsSf https://astral.sh/uv/install.sh | sh' and re-run."

    if [[ -x "$VENV_PY" && "$(installed_version ovrtx)" != "none" && "$(installed_version ovstage)" != "none" ]]; then
        return
    fi

    local reason="ovrtx/ovstage missing from $VENV_PY"
    [[ -x "$VENV_PY" ]] || reason="no interpreter at $VENV_PY"
    [[ "$BOOTSTRAP" == "1" ]] || die "$reason. Run 'uv sync --extra $UV_SYNC_EXTRAS' or unset BOOTSTRAP=0."

    log "$reason -- bootstrapping with 'uv sync --extra $UV_SYNC_EXTRAS' (first run downloads several GB)"
    local extra_args=() extra
    for extra in ${UV_SYNC_EXTRAS//,/ }; do
        extra_args+=(--extra "$extra")
    done
    uv sync "${extra_args[@]}" || die "uv sync --extra $UV_SYNC_EXTRAS failed"

    [[ -x "$VENV_PY" ]] || die "uv sync completed but $VENV_PY is still missing"
    [[ "$(installed_version ovrtx)" != "none" ]] || die \
        "uv sync completed but ovrtx is still not installed; '$UV_SYNC_EXTRAS' may be the wrong extra"
}

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
# The names are static so a typo in the arguments fails instantly, before the bootstrap sync; the
# versions behind them need $VENV_PY to read pyproject.toml, so they are resolved after it.
CONFIG_NAMES=(ovrtx_0_4_1 ovrtx_0_5_0)

# Selection: positional arguments pick a subset, in the order given.
SELECTED=("$@")
if [[ ${#SELECTED[@]} -eq 0 ]]; then
    SELECTED=("${CONFIG_NAMES[@]}")
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
# Cheap argument and input checks first, so a typo or a missing wheel fails in milliseconds rather
# than after a multi-gigabyte sync or a multi-minute run.
for name in "${SELECTED[@]}"; do
    [[ " ${CONFIG_NAMES[*]} " == *" $name "* ]] \
        || die "unknown configuration '$name'; expected one of: ${CONFIG_NAMES[*]}"
    if [[ "$name" == "ovrtx_0_5_0" && ! -f "$OVRTX_050_WHEEL" ]]; then
        die "ovrtx 0.5.0 wheel not found at $OVRTX_050_WHEEL (override with OVRTX_050_WHEEL=...)"
    fi
done

bootstrap_env

OVRTX_041_VERSION="$(pinned_version ovrtx)"
OVRTX_050_VERSION="0.5.0.0"

# name|ovrtx version|install spec
CONFIGS=(
    "ovrtx_0_4_1|$OVRTX_041_VERSION|ovrtx==$OVRTX_041_VERSION"
    "ovrtx_0_5_0|$OVRTX_050_VERSION|$OVRTX_050_WHEEL"
)

find_config() {
    local wanted="$1" entry
    for entry in "${CONFIGS[@]}"; do
        if [[ "${entry%%|*}" == "$wanted" ]]; then
            printf '%s\n' "$entry"
            return 0
        fi
    done
    return 1
}

# Restore the version pinned in pyproject.toml, not whatever happened to be installed when the
# script started. A previous run that was killed before its own restore leaves .venv on the 0.5.0
# wheel; adopting that as the baseline would make this run try to restore it from the public index,
# where it does not exist, and the repair would fail exactly when it is needed most.
BASELINE_OVRTX="$OVRTX_041_VERSION"
restore_baseline() {
    local status=$?
    if [[ "$(installed_version ovrtx)" != "$BASELINE_OVRTX" ]]; then
        log "restoring baseline ovrtx==$BASELINE_OVRTX"
        install_ovrtx_spec "ovrtx==$BASELINE_OVRTX" || log "WARNING: restore failed; run 'uv sync --extra ov' to repair .venv"
    fi
    exit "$status"
}
trap restore_baseline EXIT

TOTAL_RUNS=$((${#SELECTED[@]} * ${#MODES[@]} * REPEATS))
log "task=$TASK_NAME envs=$ENVS steps=$STEPS warmup=$WARMUP seed=$SEED"
log "ovstage=$(installed_version ovstage) (held fixed) installed ovrtx=$(installed_version ovrtx) restore target=$BASELINE_OVRTX"
log "runtimes: ${SELECTED[*]}"
log "render modes: ${MODES[*]}"
log "$TOTAL_RUNS runs total (${#SELECTED[@]} runtimes x ${#MODES[@]} modes x $REPEATS repeats)"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
FAILED=()

# The runtime is the outer loop so the wheel is swapped once per runtime rather than once per
# render mode; a swap costs a multi-gigabyte reinstall, a mode change costs nothing.
for name in "${SELECTED[@]}"; do
    entry="$(find_config "$name")"
    IFS='|' read -r _ want_version spec <<<"$entry"

    ensure_ovrtx "$want_version" "$spec"

    for mode in "${MODES[@]}"; do
        presets="$BASE_PRESETS,$mode"
        out="$DATA_ROOT/$TASK_NAME/envs_$ENVS/$mode/$name"
        mkdir -p "$out"

        # Record the runtime that produced the numbers alongside them, so a directory of results is
        # self-describing after the venv has been swapped back. The benchmark's own version_info
        # phase also records whether the working tree was dirty.
        {
            echo "config=$name"
            echo "ovrtx=$(installed_version ovrtx)"
            echo "ovstage=$(installed_version ovstage)"
            echo "ovphysx=$(installed_version ovphysx)"
            echo "task=$TASK_NAME"
            echo "presets=$presets"
            echo "camera_preset=$mode"
            echo "num_envs=$ENVS"
            echo "num_steps=$STEPS"
            echo "warmup_steps=$WARMUP"
            echo "seed=$SEED"
            echo "repeats=$REPEATS"
            echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
            echo "dirty=$([[ -n "$(git status --porcelain 2>/dev/null)" ]] && echo true || echo false)"
        } >"$out/runtime_env.txt"

        for ((run = 1; run <= REPEATS; run++)); do
            run_out="$out/run_$run"
            # Clear only this repeat, so re-running the matrix replaces its results rather than
            # accumulating timestamped JSONs that the aggregation would average together.
            rm -rf "$run_out"
            mkdir -p "$run_out"

            log "running $name/$mode repeat $run/$REPEATS (ovrtx=$(installed_version ovrtx)) -> $run_out"

            if "$VENV_PY" scripts/benchmarks/runtime.py \
                --task "$TASK_NAME" "presets=$presets" \
                --seed "$SEED" \
                --num_envs "$ENVS" \
                --num_steps "$STEPS" \
                --warmup_steps "$WARMUP" \
                --benchmark_formatter json \
                --output_path "$run_out" 2>&1 | tee "$run_out/benchmark.log"; then
                log "$name/$mode repeat $run completed"
            else
                # PIPESTATUS[0] is the benchmark's status; tee always succeeds.
                log "$name/$mode repeat $run FAILED (exit ${PIPESTATUS[0]}), see $run_out/benchmark.log"
                FAILED+=("$mode/$name/run_$run")
            fi
        done
    done
done

log "results under $DATA_ROOT/$TASK_NAME/envs_$ENVS/"

# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
# Repeats are only useful if they are reduced to a spread, so print mean +/- std across runs for
# the metrics this matrix exists to compare, one table per render mode.
"$VENV_PY" - "$DATA_ROOT/$TASK_NAME/envs_$ENVS" "$(IFS=,; echo "${MODES[*]}")" "$(IFS=,; echo "${SELECTED[*]}")" <<'PY' || log "WARNING: summary failed"
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
modes = [m for m in sys.argv[2].split(",") if m]
configs = [c for c in sys.argv[3].split(",") if c]
metrics = [
    ("Mean Total FPS", "FPS", "{:,.0f}"),
    ("Mean Iteration Time", "ms", "{:.2f}"),
    ("GPU Utilization", "%", "{:.1f}"),
    ("Scene Creation Time", "ms", "{:,.0f}"),
    ("GPU Memory Used peak", "GB", "{:.2f}"),
]


def collect(mode: str, config: str) -> dict[str, list[float]]:
    per_metric: dict[str, list[float]] = {name: [] for name, _, _ in metrics}
    for report in sorted((root / mode / config).glob("run_*/benchmark_runtime_*.json")):
        for phase in json.loads(report.read_text()):
            prefix = f"benchmark_runtime {phase['phase_name']} "
            for measurement in phase["measurements"]:
                key = measurement["name"].removeprefix(prefix)
                if key in per_metric:
                    per_metric[key].append(measurement["value"])
    return per_metric


width = max(len(name) for name, _, _ in metrics) + 8
for mode in modes:
    samples = {config: collect(mode, config) for config in configs}
    print(f"\n[ovrtx-matrix] {mode}: summary (mean +/- std over repeats)\n")
    print("metric".ljust(width) + "".join(c.rjust(28) for c in configs))
    print("-" * (width + 28 * len(configs)))
    for name, unit, fmt in metrics:
        row = f"{name} [{unit}]".ljust(width)
        for config in configs:
            values = samples[config][name]
            if not values:
                cell = "n/a"
            elif len(values) == 1:
                cell = f"{fmt.format(values[0])} (n=1)"
            else:
                cell = f"{fmt.format(statistics.mean(values))} +/- {fmt.format(statistics.stdev(values))} (n={len(values)})"
            row += cell.rjust(28)
        print(row)

    # The point of the matrix is the delta, so state it rather than leaving it to be eyeballed.
    if len(configs) == 2:
        base, cand = configs
        print(f"\n  {cand} vs {base}:")
        for name, unit, _ in metrics:
            base_values, cand_values = samples[base][name], samples[cand][name]
            if not base_values or not cand_values:
                continue
            base_mean, cand_mean = statistics.mean(base_values), statistics.mean(cand_values)
            if base_mean == 0:
                continue
            print(f"    {f'{name} [{unit}]'.ljust(width)}{(cand_mean - base_mean) / base_mean * 100:+8.2f}%")
print()
PY

if [[ ${#FAILED[@]} -gt 0 ]]; then
    die "failed runs: ${FAILED[*]}"
fi
log "all configurations completed"
