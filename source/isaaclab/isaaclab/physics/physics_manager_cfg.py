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

    The simulation launcher resolves this placeholder to Isaac Sim PhysX when
    Kit is required or when no OvPhysX alternative is configured. Otherwise,
    it resolves to the configured OvPhysX alternative for a kitless run.
    """

    class_type: Any = None
    """Placeholder manager type; resolved before simulation construction."""

    physics_type: str = "auto_physx"
    """Physics placeholder marker consumed by :func:`isaaclab.app.scan`."""

    isaacsim_physx: PhysxCfg | None = None
    """Concrete Isaac Sim PhysX configuration."""

    ovphysx: OvPhysxCfg | None = None
    """Concrete OvPhysX configuration, or ``None`` when OvPhysX is unsupported."""


def resolve_physx_auto_cfg(physics_cfg: PhysicsCfg, use_isaac_sim: bool) -> PhysicsCfg:
    """Resolve an automatic PhysX placeholder to a concrete backend.

    Args:
        physics_cfg: Physics configuration to resolve.
        use_isaac_sim: Whether the run already requires Isaac Sim and Kit.

    Returns:
        The selected concrete physics configuration. Non-automatic
        configurations are returned unchanged.

    Raises:
        ValueError: If the selected alternative has the wrong concrete type.
    """
    if not isinstance(physics_cfg, PhysxAutoCfg):
        return physics_cfg

    if not use_isaac_sim and physics_cfg.ovphysx is not None:
        from isaaclab_ovphysx.physics import OvPhysxCfg

        selected = physics_cfg.ovphysx
        expected_type = OvPhysxCfg
        field_name = "ovphysx"
    else:
        from isaaclab_physx.physics import PhysxCfg

        selected = physics_cfg.isaacsim_physx if physics_cfg.isaacsim_physx is not None else PhysxCfg()
        expected_type = PhysxCfg
        field_name = "isaacsim_physx"

    if not isinstance(selected, expected_type):
        raise ValueError(
            f"Invalid PhysxAutoCfg.{field_name}: expected {expected_type.__name__}, got {type(selected).__name__}."
        )
    return selected
