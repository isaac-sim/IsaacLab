# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments.

The package is structured as follows:

- ``direct``: These include single-file implementations of tasks.
- ``manager_based``: These include task implementations that use the manager-based API.
- ``utils``: These include utility functions for the tasks.

"""

import os

# Conveniences to other module directories via relative paths
ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

# Load extension metadata from extension.toml if available (editable install),
# otherwise fall back to importlib.metadata (pip/uv install from git).
_ext_toml_path = os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml")
if os.path.exists(_ext_toml_path):
    import toml

    ISAACLAB_TASKS_METADATA = toml.load(_ext_toml_path)
    """Extension metadata dictionary parsed from the extension.toml file."""
    __version__ = ISAACLAB_TASKS_METADATA["package"]["version"]
else:
    from importlib.metadata import metadata

    _meta = metadata("isaaclab-tasks")
    ISAACLAB_TASKS_METADATA = {"package": {"version": _meta["Version"], "description": _meta["Summary"]}}
    __version__ = _meta["Version"]

##
# Register Gym environments.
##

from .utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
# TODO(@ashwinvk): Remove pick_place from the blacklist once pinocchio from Isaac Sim is compatibility
_BLACKLIST_PKGS = ["utils", ".mdp", "pick_place", "direct.humanoid_amp.motions"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
