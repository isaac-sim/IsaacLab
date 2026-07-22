# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for physics managers."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils.configclass import configclass

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


_PhysicsPresetSelection = tuple[Callable[[], PhysicsCfg | None], str, dict[str, Any]]
_PHYSICS_PRESET_SELECTIONS: dict[int, _PhysicsPresetSelection] = {}


def _set_physics_preset_selection(
    physics_cfg: PhysicsCfg,
    preset_name: str,
    alternatives: dict[str, Any] | None = None,
) -> None:
    """Record which physics preset selected a concrete physics config."""
    physics_cfg_id = id(physics_cfg)
    try:
        physics_cfg_ref = weakref.ref(
            physics_cfg,
            lambda _ref, cfg_id=physics_cfg_id: _PHYSICS_PRESET_SELECTIONS.pop(cfg_id, None),
        )
    except TypeError:

        def physics_cfg_ref() -> PhysicsCfg:
            return physics_cfg

    physics_alternatives = {}
    if alternatives is not None and "ovphysx" in alternatives:
        physics_alternatives["ovphysx"] = alternatives["ovphysx"]
    _PHYSICS_PRESET_SELECTIONS[physics_cfg_id] = (physics_cfg_ref, preset_name, physics_alternatives)


def _get_physics_preset_metadata(physics_cfg: PhysicsCfg) -> tuple[str | None, dict[str, Any]]:
    """Return preset metadata for a concrete physics config, if any."""
    metadata = _PHYSICS_PRESET_SELECTIONS.get(id(physics_cfg))
    if metadata is None:
        return None, {}
    physics_cfg_ref, preset_name, alternatives = metadata
    if physics_cfg_ref() is not physics_cfg:
        _PHYSICS_PRESET_SELECTIONS.pop(id(physics_cfg), None)
        return None, {}
    return preset_name, alternatives


def _get_physics_preset_selection(physics_cfg: PhysicsCfg) -> str | None:
    """Return the preset name that selected a concrete physics config, if any."""
    preset_name, _ = _get_physics_preset_metadata(physics_cfg)
    return preset_name


def _resolve_auto_physx_cfg(physics_cfg: PhysicsCfg, use_isaac_sim: bool) -> PhysicsCfg:
    """Resolve the automatic ``physx`` preset selection to a concrete backend."""
    if _get_physics_preset_selection(physics_cfg) != "physx":
        return physics_cfg

    from isaaclab_physx.physics import PhysxCfg

    if not isinstance(physics_cfg, PhysxCfg):
        raise ValueError(f"Invalid physics preset 'physx': expected PhysxCfg, got {type(physics_cfg).__name__}.")
    if use_isaac_sim:
        return physics_cfg

    from isaaclab_ovphysx.physics import OvPhysxCfg

    _, alternatives = _get_physics_preset_metadata(physics_cfg)
    selected = alternatives.get("ovphysx") or OvPhysxCfg()
    if not isinstance(selected, OvPhysxCfg):
        raise ValueError(f"Invalid physics preset 'ovphysx': expected OvPhysxCfg, got {type(selected).__name__}.")
    return selected
