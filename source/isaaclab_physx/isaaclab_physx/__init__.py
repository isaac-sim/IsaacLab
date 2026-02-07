# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the PhysX simulation interfaces for IsaacLab core package."""

import os
import sys
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_PHYSX_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_PHYSX_METADATA = toml.load(os.path.join(ISAACLAB_PHYSX_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_PHYSX_METADATA["package"]["version"]


def _patch_isaacsim_simulation_manager():
    """Patch Isaac Sim's SimulationManager to use PhysxManager.

    This ensures all code that imports from isaacsim.core.simulation_manager
    will use our PhysxManager instead, preventing duplicate callback registration.
    """
    if "isaacsim.core.simulation_manager" in sys.modules:
        original_module = sys.modules["isaacsim.core.simulation_manager"]
        from .physics.physx_manager import PhysxManager, IsaacEvents

        original_module.SimulationManager = PhysxManager
        original_module.IsaacEvents = IsaacEvents


_patch_isaacsim_simulation_manager()
