# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the PhysX simulation interfaces for IsaacLab core package."""

import importlib.metadata

from ._simulation_manager_patch import _SimulationManagerPatch


try:
    __version__ = importlib.metadata.version("isaaclab_physx")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


_simulation_manager_patch = _SimulationManagerPatch()
_simulation_manager_patch.claim_physics_lifecycle()
