# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for physics managers."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .physics_manager import PhysicsManager


@configclass
class PhysicsCfg:
    """Abstract base configuration for physics managers.

    This base class contains physics backend-specific parameters.
    Subclasses should override the class_type to return the appropriate
    physics manager class.

    .. note::
        The following parameters can be optionally specified here to override
        the values from :class:`SimulationCfg`. If left as MISSING, they will
        be automatically propagated from :class:`SimulationCfg`:

        - ``dt``: Physics simulation time-step
        - ``gravity``: Gravity vector
        - ``physics_prim_path``: Physics scene prim path
        - ``physics_material``: Default physics material
    """

    # ------------------------------------------------------------------
    # Physics Backend Configuration
    # ------------------------------------------------------------------

    class_type: type[PhysicsManager] = MISSING

    # ------------------------------------------------------------------
    # Override Parameters (propagated from SimulationCfg if MISSING)
    # ------------------------------------------------------------------

    dt: float = MISSING
    """The physics simulation time-step (in seconds).

    If MISSING, uses the value from :attr:`SimulationCfg.dt`.
    """

    gravity: tuple[float, float, float] = MISSING
    """The gravity vector (in m/s^2).

    If MISSING, uses the value from :attr:`SimulationCfg.gravity`.
    """

    physics_prim_path: str = MISSING
    """The prim path where the USD PhysicsScene is created.

    If MISSING, uses the value from :attr:`SimulationCfg.physics_prim_path`.
    """

    physics_material: RigidBodyMaterialCfg = MISSING
    """Default physics material settings for rigid bodies.

    If MISSING, uses the value from :attr:`SimulationCfg.physics_material`.
    The physics engine defaults to this physics material for all the rigid body prims
    that do not have any physics material specified on them.

    The material is created at the path: ``{physics_prim_path}/defaultMaterial``.
    """
