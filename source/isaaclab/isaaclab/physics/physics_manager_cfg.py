# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for physics managers."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_ovphysx.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg

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


@configclass
class PhysxAutoCfg(PhysicsCfg):
    """Automatic PhysX-family physics placeholder.

    The simulation launcher resolves this placeholder to
    :class:`isaaclab_physx.physics.PhysxCfg` when Isaac Sim / Kit is required,
    and to :class:`isaaclab_ovphysx.physics.OvPhysxCfg` for kitless runs.
    """

    class_type: Any = None
    """Placeholder manager type; resolved before :class:`SimulationContext` construction."""

    physics_type: str = "auto_physx"
    """Physics placeholder marker consumed by :func:`isaaclab.app.scan`."""

    isaacsim_physx: PhysxCfg | None = None
    """Concrete Isaac Sim PhysX config to use when Kit is required."""

    ovphysx: OvPhysxCfg | None = None
    """Concrete OvPhysX config to use for kitless runs."""


def resolve_physx_auto_cfg(physics_cfg: PhysicsCfg, use_isaac_sim: bool) -> PhysicsCfg:
    """Resolve an automatic PhysX-family placeholder to a concrete backend.

    Args:
        physics_cfg: Physics configuration to resolve. Non-automatic configs are
            returned unchanged.
        use_isaac_sim: Whether the run requires Isaac Sim / Kit.

    Returns:
        The selected concrete physics configuration.

    Raises:
        ValueError: If a configured automatic alternative has the wrong type.
    """
    if not isinstance(physics_cfg, PhysxAutoCfg):
        return physics_cfg

    if use_isaac_sim:
        from isaaclab_physx.physics import PhysxCfg

        selected = physics_cfg.isaacsim_physx or PhysxCfg()
        expected_type = PhysxCfg
        field_name = "isaacsim_physx"
    else:
        from isaaclab_ovphysx.physics import OvPhysxCfg

        selected = physics_cfg.ovphysx or OvPhysxCfg()
        expected_type = OvPhysxCfg
        field_name = "ovphysx"

    if not isinstance(selected, expected_type):
        raise ValueError(
            f"Invalid PhysxAutoCfg.{field_name}: expected {expected_type.__name__}, got {type(selected).__name__}."
        )
    return selected
