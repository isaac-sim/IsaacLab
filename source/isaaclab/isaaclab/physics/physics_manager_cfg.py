# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for physics managers."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .physics_manager import PhysicsManager


@configclass
class PhysicsCfg:
    """Abstract base configuration for physics managers.

    This base class contains physics backend-specific parameters.
    Subclasses should override the class_type to return the appropriate
    physics manager class.

    Shared simulation parameters (dt, gravity, physics_prim_path, physics_material)
    are read directly from :class:`SimulationCfg` by the physics manager.
    """

    class_type: type[PhysicsManager] | Any = MISSING
    """The physics manager class to use. Must be set by subclasses."""
