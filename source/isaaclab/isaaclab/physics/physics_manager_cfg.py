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
    """PhysX configuration resolved to a concrete backend at launch."""

    class_type: Any = None
    """Unused because this configuration is resolved before simulation construction."""

    isaacsim_physx: PhysxCfg | None = None
    """Concrete Isaac Sim PhysX configuration, or ``None`` when unavailable."""

    ovphysx: OvPhysxCfg | None = None
    """Concrete OvPhysX configuration, or ``None`` when OvPhysX is unsupported."""


def _resolve_physx_auto_cfg(physics_cfg: PhysicsCfg, use_isaac_sim: bool) -> PhysicsCfg:
    """Resolve a :class:`PhysxAutoCfg` to a concrete backend."""
    if not isinstance(physics_cfg, PhysxAutoCfg):
        return physics_cfg

    if not use_isaac_sim and physics_cfg.ovphysx is not None:
        from isaaclab_ovphysx.physics import OvPhysxCfg

        selected = physics_cfg.ovphysx
        expected_type = OvPhysxCfg
        field_name = "ovphysx"
    else:
        from isaaclab_physx.physics import PhysxCfg

        selected = physics_cfg.isaacsim_physx
        expected_type = PhysxCfg
        field_name = "isaacsim_physx"

    if not isinstance(selected, expected_type):
        raise ValueError(
            f"Invalid PhysxAutoCfg.{field_name}: expected {expected_type.__name__}, got {type(selected).__name__}."
        )
    return selected
