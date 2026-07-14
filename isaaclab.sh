#!/usr/bin/env bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Exit on error.
set -e

# Get repo directory.
export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Find python to run CLI.
if [ -n "$VIRTUAL_ENV" ]; then
    python_exe="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ]; then
    python_exe="$CONDA_PREFIX/bin/python"
elif [ -f "$ISAACLAB_PATH/env_isaaclab/bin/python" ]; then
    python_exe="$ISAACLAB_PATH/env_isaaclab/bin/python"
elif [ -f "$ISAACLAB_PATH/_isaac_sim/python.sh" ]; then
    python_exe="$ISAACLAB_PATH/_isaac_sim/python.sh"
else
    # Fallback to system python
    python_exe="python3"
fi

# Add source/isaaclab to PYTHONPATH so we can import isaaclab.cli.
export PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$PYTHONPATH"

# Let Kit associate direct wrapper launches with the Isaac Sim desktop icon.
export RESOURCE_NAME="${RESOURCE_NAME:-IsaacSim}"

# If a local Isaac Sim binary is present, source its env setup so that
# PYTHONPATH/PATH/EXP_PATH are correct without depending on a conda
# activate.d hook (those don't fire reliably under e.g. `conda run`).
if [ -d "$ISAACLAB_PATH/_isaac_sim" ]; then
    if [ -f "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" ]; then
        # shellcheck disable=SC1091
        . "$ISAACLAB_PATH/_isaac_sim/setup_conda_env.sh" >/dev/null 2>&1 || true
    elif [ -f "$ISAACLAB_PATH/_isaac_sim/setup_python_env.sh" ]; then
        export ISAAC_PATH="$ISAACLAB_PATH/_isaac_sim"
        export CARB_APP_PATH="$ISAAC_PATH/kit"
        export EXP_PATH="$ISAAC_PATH/apps"
        # shellcheck disable=SC1091
        . "$ISAACLAB_PATH/_isaac_sim/setup_python_env.sh" >/dev/null 2>&1 || true
        # Unlike setup_conda_env.sh, setup_python_env.sh prepends Kit's
        # pip_prebundle directories to PYTHONPATH. Those ship vendored copies of
        # common libraries (e.g. an older typing_extensions lacking Sentinel)
        # that then shadow the active venv/conda environment. Put the active
        # environment's site-packages first so it always wins.
        if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_PREFIX" ]; then
            env_site_packages="$("$python_exe" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
            if [ -n "$env_site_packages" ]; then
                export PYTHONPATH="$env_site_packages:$PYTHONPATH"
            fi
        fi
    else
        echo "[WARNING] _isaac_sim is present but _isaac_sim/setup_conda_env.sh or _isaac_sim/setup_python_env.sh is missing; Isaac Sim env vars not exported." >&2
        echo "[WARNING] Re-extract the Isaac Sim binary zip if you intend to use the bundled binary." >&2
    fi
fi

# Execute CLI.
exec "$python_exe" -c "from isaaclab.cli import cli; cli()" "$@"
