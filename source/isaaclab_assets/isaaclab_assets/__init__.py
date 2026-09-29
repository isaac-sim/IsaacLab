# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing asset and sensor configurations."""

import importlib.metadata
import os

ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""


try:
    __version__ = importlib.metadata.version("isaaclab_assets")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from isaaclab.utils.module import lazy_export

lazy_export()
