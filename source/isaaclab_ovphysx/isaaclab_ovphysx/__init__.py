# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the ovphysx/TensorBindingsAPI simulation interfaces for IsaacLab."""

import os

import toml

ISAACLAB_OVPHYSX_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_OVPHYSX_METADATA = toml.load(os.path.join(ISAACLAB_OVPHYSX_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

__version__ = ISAACLAB_OVPHYSX_METADATA["package"]["version"]
