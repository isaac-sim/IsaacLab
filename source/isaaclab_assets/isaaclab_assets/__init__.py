# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing asset and sensor configurations."""

import os

# Conveniences to other module directories via relative paths
ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

# Load extension metadata from extension.toml if available (editable install),
# otherwise fall back to importlib.metadata (pip/uv install from git).
_ext_toml_path = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "config", "extension.toml")
if os.path.exists(_ext_toml_path):
    import toml

    ISAACLAB_ASSETS_METADATA = toml.load(_ext_toml_path)
    """Extension metadata dictionary parsed from the extension.toml file."""
    __version__ = ISAACLAB_ASSETS_METADATA["package"]["version"]
else:
    from importlib.metadata import metadata

    _meta = metadata("isaaclab-assets")
    ISAACLAB_ASSETS_METADATA = {"package": {"version": _meta["Version"], "description": _meta["Summary"]}}
    __version__ = _meta["Version"]

from .robots import *
from .sensors import *
