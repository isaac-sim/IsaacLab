#!/bin/bash
# Run ovphysx in standalone mode within IsaacLab environment
# WITHOUT launching IsaacSim (no LD_PRELOAD of libcarb.so)
#
# Usage: ./scripts/run_ovphysx.sh [your_script.py or -m pytest ...]
set -e

ISAACLAB_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_DIR="${ISAACLAB_PATH}/_isaac_sim"

# Source the Python environment setup (sets PYTHONPATH, LD_LIBRARY_PATH)
# but do NOT use python.sh which sets LD_PRELOAD
source "${ISAAC_DIR}/setup_python_env.sh"

# CRITICAL: Clear LD_PRELOAD to avoid Carbonite version conflict.
# python.sh sets LD_PRELOAD=$ISAAC_DIR/kit/libcarb.so which loads
# Carbonite 0.7 from IsaacSim, but ovphysx bundles Carbonite 0.8.
# Both try to tear down at process exit -> segfault.
export LD_PRELOAD=""

# Add all isaaclab source packages to PYTHONPATH so editable installs work
for pkg in isaaclab isaaclab_ovphysx isaaclab_tasks isaaclab_rl isaaclab_physx isaaclab_newton isaaclab_assets isaaclab_contrib; do
    if [ -d "${ISAACLAB_PATH}/source/${pkg}" ]; then
        export PYTHONPATH="${ISAACLAB_PATH}/source/${pkg}:${PYTHONPATH}"
    fi
done

# Use the Python binary directly
PYTHON_EXE="${ISAAC_DIR}/kit/python/bin/python3"

exec "${PYTHON_EXE}" "$@"
