# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package for externally contributed components for Isaac Lab.

This package provides externally contributed components for Isaac Lab, such as multirotors.
These components are not part of the core Isaac Lab framework yet, but are planned to be added
in the future. They are contributed by the community to extend the capabilities of Isaac Lab.
"""

import importlib.metadata
import os
import tomllib

ISAACLAB_CONTRIB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

_ext_toml = os.path.join(ISAACLAB_CONTRIB_EXT_DIR, "config", "extension.toml")
if os.path.exists(_ext_toml):
    with open(_ext_toml, "rb") as _f:
        ISAACLAB_CONTRIB_METADATA = tomllib.load(_f)
else:
    ISAACLAB_CONTRIB_METADATA = {}
"""Extension metadata dictionary parsed from the extension.toml file."""

try:
    __version__ = importlib.metadata.version("isaaclab_contrib")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
