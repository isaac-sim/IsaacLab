#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Install a cosmos-framework checkout for runtime visual domain randomization.
#
# Cosmos is always installed with --no-deps. Its dependency groups pin torch to
# match the CUDA wheel variants it publishes, which would fight Isaac Lab's own
# pin; its base dependencies do not pin torch at all. The pure-Python packages it
# imports come from the cosmos-runtime extra instead.
#
# Usage:
#   install_cosmos.sh <source> [venv]
#
# <source> is one of:
#   a local path        /path/to/cosmos-framework
#   an https git URL    https://github.com/NVIDIA/cosmos-framework.git
#   an ssh git URL      ssh://git@host:port/group/cosmos-framework.git
#
# A git source may be pinned by appending '.git@<ref>', for example
#   ssh://git@<host>:<port>/<group>/cosmos-framework.git@<ref>
#
# Re-run this after any plain `uv sync`, which removes the overlay. Passing
# `--inexact` to uv sync avoids that.

set -euo pipefail

SOURCE="${1:-https://github.com/NVIDIA/cosmos-framework.git}"
VENV="${2:-.venv}"

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "error: no virtual environment at '${VENV}'. Run 'uv sync --extra isaacsim --extra teleop' first." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export VIRTUAL_ENV="$(cd "${VENV}" && pwd)"

echo "==> installing the cosmos-runtime extra"
uv pip install -e "${REPO_ROOT}/source/isaaclab_contrib[cosmos-runtime]"

if [[ -d "${SOURCE}" ]]; then
    echo "==> installing cosmos-framework from local path ${SOURCE}"
    uv pip install --no-deps -e "${SOURCE}"
else
    # Split on '.git@' so the '@' in an ssh URL's 'git@host' is not mistaken for a
    # ref separator.
    if [[ "${SOURCE}" == *.git@* ]]; then
        URL="${SOURCE%.git@*}.git"
        REF="${SOURCE##*.git@}"
        SPEC="cosmos-framework @ git+${URL}@${REF}"
        echo "==> installing cosmos-framework from ${URL} at ${REF}"
    else
        SPEC="cosmos-framework @ git+${SOURCE}"
        echo "==> installing cosmos-framework from ${SOURCE} at its default branch"
    fi
    uv pip install --no-deps "${SPEC}"
fi

echo "==> verifying"
"${VENV}/bin/python" - <<'PY'
import importlib.metadata as md

import cosmos_framework
from cosmos_framework.inference.args import OmniSetupOverrides  # noqa: F401

version = md.version("cosmos-framework")
path = cosmos_framework.__file__
guided = False
try:
    from cosmos_framework.inference import args as _args

    guided = "guided_generation_mask" in open(_args.__file__).read()
except Exception:  # noqa: BLE001 - the probe must not fail the install
    pass

print(f"    cosmos-framework {version}")
print(f"    from {path}")
print(f"    native guided-generation API: {'yes' if guided else 'no (Isaac Lab sampler adapter)'}")
PY

cat <<'EOF'

Start with scripts/visual_dr/recipes/cosmos_nano.yaml for nvidia/Cosmos3-Nano.
For a custom four-step export, use cosmos_distilled.yaml and --checkpoint PATH.
Either framework supports mask guidance on or off. See
source/isaaclab_contrib/docs/visual_dr_setup.md for resolution and guidance settings.
EOF
