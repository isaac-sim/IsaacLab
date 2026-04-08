# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the core framework."""

import os

# Conveniences to other module directories via relative paths
ISAACLAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

# Load extension metadata from extension.toml if available (editable install),
# otherwise fall back to importlib.metadata (pip/uv install from git).
_ext_toml_path = os.path.join(ISAACLAB_EXT_DIR, "config", "extension.toml")
if os.path.exists(_ext_toml_path):
    import toml

    ISAACLAB_METADATA = toml.load(_ext_toml_path)
    """Extension metadata dictionary parsed from the extension.toml file."""
    __version__ = ISAACLAB_METADATA["package"]["version"]
else:
    from importlib.metadata import metadata

    _meta = metadata("isaaclab")
    ISAACLAB_METADATA = {"package": {"version": _meta["Version"], "description": _meta["Summary"]}}
    __version__ = _meta["Version"]
