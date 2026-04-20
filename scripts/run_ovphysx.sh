#!/bin/bash
# Run ovphysx in Kit's Python with its bundled libcarb.so preloaded.
# Use when ovphysx is installed into Kit's Python.
#
# Usage: ./scripts/run_ovphysx.sh [your_script.py or -m pytest ...]
set -e

ISAACLAB_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_DIR="${ISAACLAB_PATH}/_isaac_sim"

# Match python.sh so Kit extensions/resources resolve during training too.
export CARB_APP_PATH="${CARB_APP_PATH:-${ISAAC_DIR}/kit}"
export ISAAC_PATH="${ISAAC_PATH:-${ISAAC_DIR}}"
export EXP_PATH="${EXP_PATH:-${ISAAC_DIR}/apps}"

# Source the Python environment setup (sets PYTHONPATH, LD_LIBRARY_PATH)
# but do NOT use python.sh which sets LD_PRELOAD
source "${ISAAC_DIR}/setup_python_env.sh"

# Preload ovphysx's own libcarb.so so its Carbonite framework wins the
# SONAME race against any other libcarb.so present in the process.
_ovphysx_libcarb=""
for _sp in "${ISAAC_DIR}"/kit/python/lib/python3.*/site-packages/ovphysx/plugins/libcarb.so; do
    if [ -f "${_sp}" ]; then
        _ovphysx_libcarb="${_sp}"
        break
    fi
done
if [ -n "${_ovphysx_libcarb}" ]; then
    export LD_PRELOAD="${_ovphysx_libcarb}"
else
    export LD_PRELOAD=""
fi
unset _ovphysx_libcarb

# Ensure pxr (OpenUSD Python bindings) is on PYTHONPATH.
# setup_python_env.sh may not include the packman USD path after rebuilds.
# Search order: IsaacSim's bundled site-packages first, then the Python
# environment's own site-packages (covers pip-installed OpenUSD).
_pxr_found=false
for usd_dir in "${ISAAC_DIR}"/kit/python/lib/python3.*/site-packages/usd_core.libs/../.. \
               "${ISAAC_DIR}"/kit/python/lib/python3.*/site-packages; do
    if [ -d "${usd_dir}/pxr" ]; then
        export PYTHONPATH="${usd_dir}:${PYTHONPATH}"
        _pxr_found=true
        break
    fi
done
if [ "${_pxr_found}" = false ]; then
    # Last resort: ask Python itself where pxr lives
    _pxr_path=$("${ISAAC_DIR}/kit/python/bin/python3" -c "import pxr, os; print(os.path.dirname(os.path.dirname(pxr.__file__)))" 2>/dev/null)
    if [ -n "${_pxr_path}" ] && [ -d "${_pxr_path}/pxr" ]; then
        export PYTHONPATH="${_pxr_path}:${PYTHONPATH}"
    fi
fi
unset _pxr_found _pxr_path

# Add all isaaclab source packages to PYTHONPATH so editable installs work
for pkg in isaaclab isaaclab_ovphysx isaaclab_tasks isaaclab_rl isaaclab_physx isaaclab_newton isaaclab_assets isaaclab_contrib; do
    if [ -d "${ISAACLAB_PATH}/source/${pkg}" ]; then
        export PYTHONPATH="${ISAACLAB_PATH}/source/${pkg}:${PYTHONPATH}"
    fi
done

# Match python.sh default for Kit app resource discovery.
export RESOURCE_NAME="${RESOURCE_NAME:-IsaacSim}"

# Use the Python binary directly
PYTHON_EXE="${ISAAC_DIR}/kit/python/bin/python3"

exec "${PYTHON_EXE}" "$@"
