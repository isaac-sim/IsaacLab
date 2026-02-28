# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package providing IsaacTeleop-based teleoperation for Isaac Lab."""

import os
import toml
ISAACLAB_TELEOP_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""
ISAACLAB_TELEOP_METADATA = toml.load(os.path.join(ISAACLAB_TELEOP_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""
__version__ = ISAACLAB_TELEOP_METADATA["package"]["version"]

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
