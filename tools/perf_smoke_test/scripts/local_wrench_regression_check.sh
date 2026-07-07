#!/usr/bin/env bash
# Local (native, no Docker) confirmation of the WrenchComposer zero-range regression
# (#5265 introduced it, #5688 fixed it via the apply_external_force_torque early-return guard).
#
# Question: does removing the guard regress Newton more than PhysX on
# Isaac-Velocity-Flat-G1-v0? Mirrors the perf-smoke verdict (newton BLOCK, physx PASS).
# This machine is aarch64 + L40S; the gate currently runs on RTX 6000, so this is a
# cross-GPU sanity check on the *ordering* (newton hit harder than physx), not a direct
# FPS reproduction. (Note: tasks.json still labels runs_on: gpu-l40s -- stale.)
#
# Runs each backend twice -- guard present ("guarded") vs removed ("unguarded") -- via the
# native uv environment, then prints mean "Environment step effective FPS" (warmup frames
# excluded, matching tasks.json excluded_frames=[[0,100]]) and the per-backend % delta.
#
# Requires a venv at $REPO/.venv with: the uv project synced (`uv sync --extra ov --extra rtx`)
# for Newton, plus full Isaac Sim kit for PhysX (`uv pip install "isaacsim[all,extscache]==6.0.0.1"`).
# Runs via the venv python directly (uv run would reconcile away the Kit install). From repo root:
#   tools/perf_smoke_test/scripts/local_wrench_regression_check.sh
#
# Tunables via env: REPO BACKENDS NUM_ENVS NUM_FRAMES SEED REPEATS EXCLUDE OUT_ROOT
set -euo pipefail

REPO="${REPO:-/home/horde/dev/IsaacLab-fork}"
TASK="Isaac-Velocity-Flat-G1-v0"
BACKENDS="${BACKENDS:-newton}"          # space-separated: "newton physx"
NUM_ENVS="${NUM_ENVS:-512}"
NUM_FRAMES="${NUM_FRAMES:-300}"
SEED="${SEED:-42}"
REPEATS="${REPEATS:-2}"
EXCLUDE="${EXCLUDE:-100}"               # warmup frames excluded from the FPS mean
OUT_ROOT="${OUT_ROOT:-/tmp/l40s_wrench_check}"
EVENTS="$REPO/source/isaaclab/isaaclab/envs/mdp/events.py"
PYBIN="${PYBIN:-$REPO/.venv/bin/python}"   # run via venv python directly (NOT `uv run`,
                                           # which would reconcile away the Kit install)
export OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ISAACLAB_PATH="$REPO"
# aarch64 Kit requires libgomp preloaded (TLS allocation ordering); harmless for Newton.
export LD_PRELOAD="${LD_PRELOAD:-/lib/aarch64-linux-gnu/libgomp.so.1}"

cd "$REPO"
mkdir -p "$OUT_ROOT"

remove_guard() {
  python3 - "$EVENTS" <<'PY'
import sys
p = sys.argv[1]; src = open(p).read()
block = (
    "    # Skip force application if the wrench ranges are zero\n"
    "    if force_range[0] == 0.0 and force_range[1] == 0.0 and torque_range[0] == 0.0 and torque_range[1] == 0.0:\n"
    "        return\n\n"
)
if block not in src:
    sys.exit("guard block not found verbatim; aborting so we don't corrupt the file")
open(p, "w").write(src.replace(block, "", 1))
PY
}
restore_guard() { git -C "$REPO" checkout -- "$EVENTS"; }
trap restore_guard EXIT

mean_fps() {  # <json> <exclude>
  python3 - "$1" "$2" <<'PY'
import json, sys, statistics
def find(o, key="Environment step effective FPS"):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key and isinstance(v, list): return v
            r = find(v, key)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = find(v, key)
            if r is not None: return r
    return None
try:
    s = find(json.load(open(sys.argv[1])))
    ex = int(sys.argv[2])
    s = s[ex:] if s and len(s) > ex else s
    print(f"{statistics.mean(s):.1f}" if s else "NA")
except Exception:
    print("NA")
PY
}

run_one() {  # <state> <backend> <rep>
  local state="$1" backend="$2" rep="$3"
  local hydra=""; [ "$backend" = newton ] && hydra="physics=newton_mjwarp"
  local outdir="$OUT_ROOT/$state/$backend/rep$rep"; mkdir -p "$outdir"; rm -f "$outdir"/*.json
  echo "[run] state=$state backend=$backend rep=$rep" >&2
  "$PYBIN" scripts/benchmarks/benchmark_non_rl.py \
    --task "$TASK" --num_envs "$NUM_ENVS" --num_frames "$NUM_FRAMES" \
    --benchmark_backend JSONFileMetrics --output_path "$outdir" --seed "$SEED" \
    --headless $hydra > "$outdir/console.log" 2>&1 || echo "[run] WARNING nonzero exit (see $outdir/console.log)" >&2
  local j; j="$(ls -t "$outdir"/benchmark_non_rl_*.json 2>/dev/null | head -1 || true)"
  [ -n "$j" ] && mean_fps "$j" "$EXCLUDE" || echo "NA"
}

declare -A FPS
for backend in $BACKENDS; do
  restore_guard
  for r in $(seq 1 "$REPEATS"); do FPS["guarded,$backend,$r"]=$(run_one guarded "$backend" "$r"); done
  remove_guard
  for r in $(seq 1 "$REPEATS"); do FPS["unguarded,$backend,$r"]=$(run_one unguarded "$backend" "$r"); done
  restore_guard
done

avg() { python3 -c "import sys,statistics; xs=[float(x) for x in sys.argv[1:] if x not in ('NA','')]; print(f'{statistics.mean(xs):.1f}' if xs else 'NA')" "$@"; }

echo
echo "============== L40S (aarch64) WrenchComposer regression check =============="
printf "%-8s | %-12s | %-14s | %-10s\n" backend guarded_fps unguarded_fps delta_pct
echo "---------------------------------------------------------------------------"
for backend in $BACKENDS; do
  g=(); u=()
  for r in $(seq 1 "$REPEATS"); do g+=("${FPS[guarded,$backend,$r]}"); u+=("${FPS[unguarded,$backend,$r]}"); done
  gm=$(avg "${g[@]}"); um=$(avg "${u[@]}")
  d=$(python3 -c "g='$gm'; u='$um'; print(f'{(float(u)-float(g))/float(g)*100:+.2f}%' if g not in ('NA','') and u not in ('NA','') and float(g) else 'NA')")
  printf "%-8s | %-12s | %-14s | %-10s\n" "$backend" "$gm" "$um" "$d"
done
echo "==========================================================================="
echo "Raw per-run output + console logs under: $OUT_ROOT"
