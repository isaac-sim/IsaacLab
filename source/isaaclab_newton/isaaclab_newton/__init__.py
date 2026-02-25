# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the Newton simulation interfaces for IsaacLab core package."""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_NEWTON_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_NEWTON_METADATA = toml.load(os.path.join(ISAACLAB_NEWTON_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_NEWTON_METADATA["package"]["version"]
