#!/usr/bin/env bash
# Benchmark Newton visualizer loading: baseline (SDP builds from USD) vs cloner prebuild.
#
# Requires: PhysX physics backend + Newton visualizer (cartpole env).
# Set ISAACLAB_PROFILE_NEWTON_VIS_BUILD=1 to emit timing; this script sets it.
#
# Usage:
#   ./scripts/mtrepte/run_newton_vis_loading_benchmark.sh [--num_envs N] [--isaaclab path]
#
# Example:
#   ./scripts/mtrepte/run_newton_vis_loading_benchmark.sh --num_envs 4096

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ISAACLAB="${ROOT_DIR}/isaaclab.sh"
NUM_ENVS=4096

while [[ $# -gt 0 ]]; do
  case $1 in
    --num_envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --isaaclab)
      ISAACLAB="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

BENCH_PY_REL="scripts/mtrepte/run_newton_vis_loading_benchmark.py"
BENCH_PY_ABS="${ROOT_DIR}/${BENCH_PY_REL}"
if [[ ! -f "$BENCH_PY_ABS" ]]; then
  echo "Benchmark script not found: $BENCH_PY_ABS" >&2
  exit 1
fi

if [[ ! -x "$ISAACLAB" ]]; then
  echo "Isaac Lab launcher not found or not executable: $ISAACLAB" >&2
  exit 1
fi

# Run from repo root so -p path is correct
cd "$ROOT_DIR"

# Log line emitted by PhysxSceneDataProvider when ISAACLAB_PROFILE_NEWTON_VIS_BUILD=1
# Example: ... Newton model build source=usd_fallback num_envs=4096 elapsed_ms=123.45
parse_line() {
  local out="$1"
  local source
  local elapsed
  source=$(echo "$out" | grep -oE "Newton model build source=[^ ]+" | sed 's/Newton model build source=//')
  elapsed=$(echo "$out" | grep -oE "elapsed_ms=[0-9.]+" | sed 's/elapsed_ms=//')
  if [[ -n "$source" && -n "$elapsed" ]]; then
    echo "${source}\t${elapsed}"
  else
    echo ""
  fi
}

echo "Running Newton visualizer loading benchmark (PhysX + Newton viz, num_envs=${NUM_ENVS})..."
echo ""

# 1) Baseline: force SDP to build from USD (no prebuilt artifact)
echo "  [1/2] Baseline (SDP builds Newton model from USD)..."
OUT_BASELINE=$(ISAACLAB_NEWTON_VIS_USE_PREBUILT=0 \
  "$ISAACLAB" -p "$BENCH_PY_REL" --headless --visualizer newton --num_envs "$NUM_ENVS" --benchmark_steps 2 2>&1)
BASELINE=$(parse_line "$OUT_BASELINE")
if [[ -z "$BASELINE" ]]; then
  echo "  Warning: could not parse baseline timing from log (missing Newton model build line?)."
  BASELINE="usd_fallback	?"
fi

# 2) With cloner: use prebuilt artifact (default)
echo "  [2/2] With cloner (SDP uses prebuilt Newton artifact)..."
OUT_CLONER=$("$ISAACLAB" -p "$BENCH_PY_REL" --headless --visualizer newton --num_envs "$NUM_ENVS" --benchmark_steps 2 2>&1)
CLONER=$(parse_line "$OUT_CLONER")
if [[ -z "$CLONER" ]]; then
  echo "  Warning: could not parse cloner timing from log."
  CLONER="prebuilt_cloner_artifact	?"
fi

echo ""
echo "----------------------------------------"
echo "  Newton model build (SDP) timing (ms)"
echo "----------------------------------------"
printf  "  %-28s %s\n" "Method" "elapsed_ms"
echo "  --------------------------------------"
printf  "  %-28s %s\n" "Baseline (USD fallback)" "$(echo "$BASELINE" | cut -f2)"
printf  "  %-28s %s\n" "With cloner (prebuilt)"  "$(echo "$CLONER" | cut -f2)"
echo "----------------------------------------"
echo ""
echo "Done. num_envs=$NUM_ENVS"
