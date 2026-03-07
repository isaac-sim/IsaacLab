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

# Enable provider timing output unless caller already set it explicitly.
export ISAACLAB_PROFILE_NEWTON_VIS_BUILD="${ISAACLAB_PROFILE_NEWTON_VIS_BUILD:-1}"

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
  local parsed
  parsed=$(printf "%s\n" "$out" | sed -n 's/.*Newton model build source=\([^ ]*\).*elapsed_ms=\([0-9.]*\).*/\1 \2/p' | sed -n '$p')
  if [[ -n "$parsed" ]]; then
    printf "%s\t%s\n" "${parsed%% *}" "${parsed#* }"
  else
    echo ""
  fi
}

# Run one benchmark leg and capture both output and total wall-clock elapsed time (ms).
run_case() {
  local use_prebuilt="$1"
  local out_var="$2"
  local elapsed_var="$3"

  local start_ms end_ms out
  start_ms=$(date +%s%3N)
  out=$(ISAACLAB_NEWTON_VIS_USE_PREBUILT="$use_prebuilt" \
    "$ISAACLAB" -p "$BENCH_PY_REL" --headless --visualizer newton --num_envs "$NUM_ENVS" --benchmark_steps 2 2>&1)
  end_ms=$(date +%s%3N)

  printf -v "$out_var" "%s" "$out"
  printf -v "$elapsed_var" "%s" "$((end_ms - start_ms))"
}

echo "Running Newton visualizer loading benchmark (PhysX + Newton viz, num_envs=${NUM_ENVS})..."
echo ""

# 1) Baseline: force SDP to build from USD (no prebuilt artifact)
echo "  [1/2] Baseline (SDP builds Newton model from USD)..."
run_case 0 OUT_BASELINE BASELINE_TOTAL_MS
BASELINE=$(parse_line "$OUT_BASELINE")
if [[ -z "$BASELINE" ]]; then
  echo "  Warning: could not parse baseline timing from log (missing Newton model build line?)."
  BASELINE="usd_fallback	?"
fi

# 2) With cloner: use prebuilt artifact (default)
echo "  [2/2] With cloner (SDP uses prebuilt Newton artifact)..."
run_case 1 OUT_CLONER CLONER_TOTAL_MS
CLONER=$(parse_line "$OUT_CLONER")
if [[ -z "$CLONER" ]]; then
  echo "  Warning: could not parse cloner timing from log."
  CLONER="prebuilt_cloner_artifact	?"
fi

echo ""
echo "---------------------------------------------------"
echo "  Total process wall time (includes cloner prebuild)"
echo "---------------------------------------------------"
printf  "  %-28s %s\n" "Method" "elapsed_ms"
echo "  --------------------------------------"
printf  "  %-28s %s\n" "Baseline (USD fallback)" "$BASELINE_TOTAL_MS"
printf  "  %-28s %s\n" "With cloner (prebuilt)"  "$CLONER_TOTAL_MS"
echo "---------------------------------------------------"
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
