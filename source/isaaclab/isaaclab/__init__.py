# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the core framework."""

import os

# Conveniences to other module directories via relative paths.
ISAACLAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

# The CLI imports this module to run installation. We must handle the case where
# dependencies (like toml) are not yet installed in a fresh environment.
# This prevents ImportError during the initial bootstrap phase.
try:
    import toml
    ISAACLAB_METADATA = toml.load(os.path.join(ISAACLAB_EXT_DIR, "config", "extension.toml"))
    """Extension metadata dictionary parsed from the extension.toml file."""
    __version__ = ISAACLAB_METADATA["package"]["version"]
except ImportError:
    # Check for tomllib (Python 3.11+).
    try:
        import tomllib
        with open(os.path.join(ISAACLAB_EXT_DIR, "config", "extension.toml"), "rb") as f:
            ISAACLAB_METADATA = tomllib.load(f)
        __version__ = ISAACLAB_METADATA["package"]["version"]
    except (ImportError, FileNotFoundError):
        # Tomllib is not part of the standard library before Python 3.11.
        # Stub is good enough for installation purposes.
        ISAACLAB_METADATA = {"package": {"version": "0.0.0"}}
        __version__ = "0.0.0"
