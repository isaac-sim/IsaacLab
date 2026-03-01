# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing utilities for common operations and helper functions."""

import os as _os
import re as _re

from .configclass import configclass

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)


def _parse_kit_asset_root() -> str:
    """Parse ``persistent.isaac.asset_root.cloud`` from ``apps/isaaclab.python.kit``."""
    root = _os.path.join(_os.path.dirname(__file__), *([".."] * 4))
    kit_path = _os.path.normpath(_os.path.join(root, "apps", "isaaclab.python.kit"))
    with open(kit_path) as f:
        for line in reversed(f.readlines()):
            m = _re.match(r'\s*persistent\.isaac\.asset_root\.cloud\s*=\s*"([^"]*)"', line)
            if m:
                return m.group(1)
    return ""


NUCLEUS_ASSET_ROOT_DIR: str = _parse_kit_asset_root()
"""Path to the root directory on the Nucleus Server."""

NVIDIA_NUCLEUS_DIR: str = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
"""Path to the root directory on the NVIDIA Nucleus Server."""

ISAAC_NUCLEUS_DIR: str = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""

ISAACLAB_NUCLEUS_DIR: str = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the ``Isaac/IsaacLab`` directory on the NVIDIA Nucleus Server."""
