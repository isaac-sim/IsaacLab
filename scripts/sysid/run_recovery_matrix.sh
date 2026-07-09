#!/bin/bash
# Synthetic recovery matrix runner (frozen surface, stop-on-failure).
#
# Cells: {nominal, hot} x seeds {0,1} = MECHANISM evidence, plus an optional
# PROTOCOL cell replaying the exact command matrix of a collected dataset
# (PROTOCOL_DATASET env var) through known gains — the evidence that drives
# the excitation-redesign decision.
#
# Status: $RESULT_DIR/matrix_result.json is updated ATOMICALLY (tmp+mv) with
# state (running|dry_run|pass|fail), current_cell, and per-cell results. The
# runner STOPS at the first failed cell (exit nonzero); an aggregate pass
# requires every executed cell to pass its 5% MAPE recovery gate.
#
# Usage: run_recovery_matrix.sh [RESULT_DIR] [--dry-run]
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
RESULT_DIR=${1:-/tmp/fr3_recovery_matrix}
DRY_RUN=${2:-}
P=${SYSID_PYTHON:-python3}
KP="1500 1500 1500 1250 1250 1000 1000"
KD="55 50 48 40 25 18 12"
STATUS=$RESULT_DIR/matrix_result.json
mkdir -p "$RESULT_DIR"

cd "$REPO"
export PYTHONPATH=$(ls -d "$REPO"/source/*/ | paste -sd:):${PYTHONPATH:-}
export OMNI_KIT_ACCEPT_EULA=YES

status_update() {  # status_update <python-dict-expr-applied-to-d>
    $P - "$STATUS" "$1" <<'PYEOF'
import json, os, sys
path, expr = sys.argv[1], sys.argv[2]
d = json.load(open(path)) if os.path.exists(path) else {"state": "init", "runs": {}, "pass": False}
exec(expr)
tmp = path + ".tmp"
json.dump(d, open(tmp, "w"), indent=2)
os.replace(tmp, path)
PYEOF
}

CELLS="nominal_s0 nominal_s1 hot_s0 hot_s1"
if [ -n "${PROTOCOL_DATASET:-}" ]; then CELLS="$CELLS protocol_s0"; fi

# Every invocation atomically RESETS the status — a rerun over a prior pass
# must never retain stale runs/pass=true.
RESET="d.clear(); d['runs']={}; d['pass']=False; d['current_cell']=None; d['cells']='$CELLS'"

if [ "$DRY_RUN" = "--dry-run" ]; then
    status_update "$RESET; d['state']='dry_run'"
    echo "DRY RUN — cells: $CELLS; status at $STATUS"
    exit 0
fi

status_update "$RESET; d['state']='running'"

# Non-cell failures (dataset generation, unexpected set -e exits) must not
# leave state=running behind.
on_exit() {
    rc=$?
    if [ "$rc" -ne 0 ]; then
        status_update "d['state']='fail'; d['pass']=False" || true
    fi
}
trap on_exit EXIT

echo "=== $(date) generating datasets ==="
status_update "d['current_cell']='datasets'"
$P -u scripts/sysid/make_synthetic_dataset.py --out "$RESULT_DIR"/data_nominal/chirp_data.pt \
    --duration 20 --scale 0.10 --f_min 0.3 --f_max 2.0 \
    --stiffness $KP --damping $KD --visualizer none
$P -u scripts/sysid/make_synthetic_dataset.py --out "$RESULT_DIR"/data_hot/chirp_data.pt \
    --duration 20 --scale 0.12 --f_min 0.5 --f_max 3.0 \
    --stiffness $KP --damping $KD --visualizer none
if [ -n "${PROTOCOL_DATASET:-}" ]; then
    $P -u scripts/sysid/make_synthetic_dataset.py --out "$RESULT_DIR"/data_protocol/chirp_data.pt \
        --from_dataset "$PROTOCOL_DATASET" --stiffness $KP --damping $KD --visualizer none
fi

for cell in $CELLS; do
    level=${cell%_s*}
    seed=${cell##*_s}
    echo "=== $(date) fit $cell ==="
    status_update "d['current_cell']='$cell'; d['runs'].setdefault('$cell', {})['state']='running'"
    if ! $P -u scripts/sysid/fit.py --data "$RESULT_DIR"/data_$level/chirp_data.pt \
        --num_envs 256 --seed "$seed" --log_dir "$RESULT_DIR"/logs_$cell \
        --plot_script '' --visualizer none > "$RESULT_DIR"/fit_$cell.log 2>&1; then
        status_update "d['runs']['$cell']={'state':'fit_failed'}; d['state']='fail'"
        echo "FIT $cell FAILED — stopping (see $RESULT_DIR/fit_$cell.log)"
        exit 1
    fi
    run_dir=$(ls -dt "$RESULT_DIR"/logs_$cell/*/ | head -1)
    if ! $P scripts/sysid/recovery_report.py --data "$RESULT_DIR"/data_$level/chirp_data.pt \
        --run "$run_dir" --max_mape 5.0 > "$RESULT_DIR"/report_$cell.log 2>&1; then
        status_update "d['runs']['$cell']={'state':'recovery_failed','run_dir':'$run_dir'}; d['state']='fail'"
        echo "RECOVERY $cell FAILED — stopping (see $RESULT_DIR/report_$cell.log)"
        exit 1
    fi
    tail -3 "$RESULT_DIR"/report_$cell.log
    status_update "d['runs']['$cell']={'state':'pass','run_dir':'$run_dir'}"
done

status_update "d['state']='pass'; d['pass']=True; d['current_cell']=None"
echo "=== $(date) MATRIX PASS — $STATUS ==="
