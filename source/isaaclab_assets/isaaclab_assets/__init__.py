# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing asset and sensor configurations."""

from __future__ import annotations

import os
import toml
ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""
ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""
ISAACLAB_ASSETS_METADATA = toml.load(os.path.join(ISAACLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""
__version__ = ISAACLAB_ASSETS_METADATA["package"]["version"]

import typing

if typing.TYPE_CHECKING:
    from .robots import *  # noqa: F403
    from .sensors import *  # noqa: F403

from isaaclab.utils.module import cascading_export

cascading_export(
    submodules=["robots", "sensors"],
)
