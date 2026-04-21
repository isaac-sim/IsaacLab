# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the ovphysx/TensorBindingsAPI simulation interfaces for IsaacLab."""

import os

import toml

ISAACLAB_OVPHYSX_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

# Find config/extension.toml: bundled inside the package (wheel install) or in the
# parent directory (editable install).
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_toml_path = os.path.join(_pkg_dir, "config", "extension.toml")
if not os.path.isfile(_toml_path):
    _toml_path = os.path.join(ISAACLAB_OVPHYSX_EXT_DIR, "config", "extension.toml")

ISAACLAB_OVPHYSX_METADATA = toml.load(_toml_path)
"""Extension metadata dictionary parsed from the extension.toml file."""

__version__ = ISAACLAB_OVPHYSX_METADATA["package"]["version"]
