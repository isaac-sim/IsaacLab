#!/usr/bin/env bash
# Cartpole + Shadow Hand Repose: perspective and tiled video per preset.
# PhysX + newton_renderer is omitted from PRESETS (Newton Warp obs cams fail at reset).
# Continues on per-run failure (e.g. PhysX+OVRTX) so other combinations still run.
#
# Usage from IsaacLab repo root:
#   ./scripts/benchmarks/run_benchmark_presets.sh

set -e

CONDA_ENV_NAME="${CONDA_ENV_NAME:-my_isaaclab_env}"
for _base in "$HOME/miniconda/envs" "$HOME/miniconda3/envs"; do
  if [[ -x "$_base/$CONDA_ENV_NAME/bin/python" ]]; then
    export CONDA_PREFIX="$_base/$CONDA_ENV_NAME"
    break
  fi
done
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[ERROR] Python not found for env $CONDA_ENV_NAME. Set CONDA_PREFIX manually."
  exit 1
fi
echo "[INFO] Using CONDA_PREFIX=$CONDA_PREFIX"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ISAACLAB_SH="${REPO_ROOT}/isaaclab.sh"
BENCHMARK_PY="${REPO_ROOT}/scripts/benchmarks/benchmark_non_rl.py"

PRESETS=(
  "physx,isaacsim_rtx_renderer,rgb"
  # "physx,newton_renderer,rgb"
  "physx,ovrtx_renderer,rgb"
  "newton,isaacsim_rtx_renderer,rgb"
  "newton,newton_renderer,rgb"
  "newton,ovrtx_renderer,rgb"
)

VIDEO_MODES=(perspective tiled)

CARTPOLE_TASK="Isaac-Cartpole-Camera-Presets-Direct-v0"
SHADOW_TASK="Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0"

cd "$REPO_ROOT"

run_benchmark() {
  local task="$1"
  local vmode="$2"
  local presets="$3"
  "$ISAACLAB_SH" -p "$BENCHMARK_PY" \
    --task="$task" \
    --num_envs=2 \
    --num_frames=2 \
    --headless \
    --enable_cameras \
    --video \
    --video_mode="$vmode" \
    "presets=$presets"
}

ANY_FAIL=0
set +e
for presets in "${PRESETS[@]}"; do
  for vmode in "${VIDEO_MODES[@]}"; do
    echo "========== Cartpole  video_mode=$vmode  presets=$presets =========="
    run_benchmark "$CARTPOLE_TASK" "$vmode" "$presets"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[WARN] Cartpole failed video_mode=$vmode presets=$presets (exit $rc)"
      ANY_FAIL=1
    fi

    echo "========== Shadow Hand  video_mode=$vmode  presets=$presets =========="
    run_benchmark "$SHADOW_TASK" "$vmode" "$presets"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[WARN] Shadow Hand failed video_mode=$vmode presets=$presets (exit $rc)"
      ANY_FAIL=1
    fi
  done
done
set -e

echo "========== All preset runs finished =========="
echo "[INFO] Videos: $REPO_ROOT/benchmark/$CARTPOLE_TASK/<timestamp>/videos/"
echo "[INFO] Videos: $REPO_ROOT/benchmark/$SHADOW_TASK/<timestamp>/videos/"
exit "$ANY_FAIL"
