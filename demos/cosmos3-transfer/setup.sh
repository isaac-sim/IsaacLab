#!/usr/bin/env bash
# Set up the Shadow Hand Cosmos demo: ./demos/cosmos3-transfer/setup.sh --bootstrap | --access | --check | --all
# --bootstrap installs uv and the Hugging Face CLI (no sudo) and prints the apt packages still needed.
# --access walks you through the Hugging Face token, the model license, repo access and the Isaac Sim license (yes/no).
# --check  checks the machine and your access to every download (reads sources.json).
# --all    installs everything (Cosmos Framework, simulator environment, checkpoint) after the check.
# --versions shows every source, access level and pinned snapshot, and what is present.
set -euo pipefail
DEMOS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DEMOS/../.."
case "${1:-}" in
  --bootstrap) shift; exec python3 "$DEMOS/demo_ctl.py" bootstrap "$@" ;;
  --access) shift; exec python3 "$DEMOS/demo_ctl.py" access "$@" ;;
  --accept-eula) shift; exec python3 "$DEMOS/demo_ctl.py" access --accept-eula "$@" ;;
  --versions) shift; exec python3 "$DEMOS/demo_ctl.py" versions "$@" ;;
  --check) shift; exec python3 "$DEMOS/demo_ctl.py" check "$@" ;;
  --all) shift; exec python3 "$DEMOS/demo_ctl.py" setup "$@" ;;
  *) echo "usage: setup.sh --bootstrap | --access | --check | --all | --versions" >&2; exit 2 ;;
esac
