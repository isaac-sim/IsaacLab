# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing multirotor (drone) support for Isaac Lab."""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_CONTRIB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_CONTRIB_METADATA = toml.load(os.path.join(ISAACLAB_CONTRIB_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_CONTRIB_METADATA["package"]["version"]
