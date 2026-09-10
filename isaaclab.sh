#!/usr/bin/env bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Exit on error.
set -e

# Get repo directory.
export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Downloaded Isaac Sim packages must run with their bundled Python. Live source
# builds created by --isaacsim_source carry a marker and support active environments.
isaacsim_source_requested=false
for arg in "$@"; do
    if [ "$arg" = "--isaacsim_source" ]; then
        isaacsim_source_requested=true
        break
    fi
done

# A virtual environment reuses the interpreter it was created from, so one created on the
# package's own Python runs that exact binary and loads Kit's extensions unchanged. Only
# environments that supply their own interpreter, such as conda, are rejected. Anything
# unreadable counts as foreign so the check fails closed.
venv_on_bundled_python=false
if [ -n "$VIRTUAL_ENV" ] && [ -z "$CONDA_PREFIX" ] && [ -f "$VIRTUAL_ENV/pyvenv.cfg" ]; then
    venv_home=$(sed -n 's/^[[:space:]]*home[[:space:]]*=[[:space:]]*//p' "$VIRTUAL_ENV/pyvenv.cfg" | head -1)
    sim_root=$(readlink -f "$ISAACLAB_PATH/_isaac_sim" 2>/dev/null)
    venv_home=$(readlink -f "$venv_home" 2>/dev/null)
    if [ -n "$sim_root" ] && [ -n "$venv_home" ]; then
        case "$venv_home" in
            "$sim_root" | "$sim_root"/*) venv_on_bundled_python=true ;;
        esac
    fi
fi

downloaded_isaac_sim=false
if [ -d "$ISAACLAB_PATH/_isaac_sim" ] && [ ! -f "$ISAACLAB_PATH/_isaac_sim/.isaaclab_source_build" ]; then
    downloaded_isaac_sim=true
    if [ "$isaacsim_source_requested" = false ] && [ "$venv_on_bundled_python" = false ] \
        && { [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_PREFIX" ]; }; then
        echo "[ERROR] Downloaded Isaac Sim packages cannot be combined with a Python virtual environment." >&2
        echo "[ERROR] Use the bundled Python after deactivating the virtual environment, create the environment on that Python ('uv venv --python \$ISAACLAB_PATH/_isaac_sim/kit/python/bin/python3'), or run '--isaacsim_source PATH' to link a live source build." >&2
        exit 1
    fi
fi

# Find python to run CLI.
if [ -n "$VIRTUAL_ENV" ]; then
    python_exe="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ]; then
    python_exe="$CONDA_PREFIX/bin/python"
elif [ "$downloaded_isaac_sim" = false ] && [ -f "$ISAACLAB_PATH/env_isaaclab/bin/python" ]; then
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
        # setup_python_env.sh also adds Kit's bundled Python stdlib
        # (kit/python/lib/python3.12) which shadows the active interpreter's
        # stdlib; its platform.py cannot parse conda-forge sys.version strings.
        export PYTHONPATH="$(echo "$PYTHONPATH" | tr ':' '\n' | grep -vE '/kit/python/lib/python3\.[0-9]+$' | paste -sd:)"
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
