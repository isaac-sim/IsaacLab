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
elif [ -f "$ISAACLAB_PATH/_isaac_sim/python.sh" ]; then
    python_exe="$ISAACLAB_PATH/_isaac_sim/python.sh"
else
    # Fallback to system python
    python_exe="python3"
fi

# Add source/isaaclab to PYTHONPATH so we can import isaaclab.cli.
export PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$PYTHONPATH"

# On Linux, preload libcarb.so so the dynamic linker reserves its static-TLS
# slot before other large native extensions (e.g. PyTorch) exhaust the surplus.
# Without this, importing omni.client later can fail with:
#   "cannot allocate memory in static TLS block"
if [ "$(uname -s)" = "Linux" ]; then
    _libcarb="$("$python_exe" -c "
import sys, os
for p in sys.path:
    f = os.path.join(p, 'omni', 'client', 'libcarb.so')
    if os.path.isfile(f):
        print(f)
        break
" 2>/dev/null)"
    if [ -n "$_libcarb" ] && [ -r "$_libcarb" ]; then
        case ":${LD_PRELOAD:-}:" in
            *":$_libcarb:"*) ;;
            *) export LD_PRELOAD="${_libcarb}${LD_PRELOAD:+:$LD_PRELOAD}";;
        esac
    fi
    unset _libcarb
fi

# Execute CLI.
exec "$python_exe" -c "from isaaclab.cli import cli; cli()" "$@"
