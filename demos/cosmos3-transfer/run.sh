#!/usr/bin/env bash
# Run the Shadow Hand Cosmos demo: ./demos/cosmos3-transfer/run.sh [--sim-gpu N] [--cosmos-gpu M]
set -euo pipefail
DEMOS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEMOS/../.."
command -v uv >/dev/null || { echo "uv is not installed or not on PATH. Run: ./demos/cosmos3-transfer/setup.sh --bootstrap" >&2; exit 1; }
# nvidia-ml-py lets the report read GPU memory on MIG hosts, where nvidia-smi hides it. Optional: skipped if it cannot be fetched.
NVML=(--with nvidia-ml-py==13.595.45)
uv run --no-sync "${NVML[@]}" python -c "import pynvml" >/dev/null 2>&1 || NVML=()
exec uv run --no-sync "${NVML[@]}" python "$DEMOS/demo_ctl.py" run "$@"
