# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for physics managers."""

from __future__ import annotations

import re
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, cast

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg

    from .physics_manager import PhysicsManager


@configclass
class CollisionGroupCfg:
    """Declarative membership and filtering rules for one collision group.

    Prim selectors are regular expressions matched against the entire prim path. A group filters
    collisions with the groups named by :attr:`filtered_groups`. When
    :attr:`invert_filtered_groups` is set, it instead filters collisions with every group not
    named by :attr:`filtered_groups`.
    """

    prim_path_exprs: tuple[str, ...] = cast(tuple[str, ...], MISSING)
    """Whole-path regular expressions selecting collider prims in this group."""

    filtered_groups: tuple[str, ...] = ()
    """Names of collision groups used by this group's filtering rule."""

    invert_filtered_groups: bool = False
    """Whether :attr:`filtered_groups` is the allow-list rather than the deny-list."""

    def validate_config(self) -> None:
        """Validate the selector regular expressions."""
        if not isinstance(self.prim_path_exprs, (list, tuple)):
            raise TypeError("CollisionGroupCfg.prim_path_exprs must be a list or tuple of strings.")
        if not all(isinstance(selector, str) for selector in self.prim_path_exprs):
            raise TypeError("CollisionGroupCfg.prim_path_exprs must contain only strings.")
        if not isinstance(self.filtered_groups, (list, tuple)):
            raise TypeError("CollisionGroupCfg.filtered_groups must be a list or tuple of strings.")
        if not all(isinstance(group_name, str) for group_name in self.filtered_groups):
            raise TypeError("CollisionGroupCfg.filtered_groups must contain only strings.")
        if not isinstance(self.invert_filtered_groups, bool):
            raise TypeError("CollisionGroupCfg.invert_filtered_groups must be a bool.")
        for selector in self.prim_path_exprs:
            try:
                re.compile(selector)
            except (re.error, TypeError) as exc:
                raise ValueError(f"Invalid collision-group prim-path regex {selector!r}: {exc}.") from exc


@configclass
class CollisionFilterCfg:
    """Backend-neutral declarative collision filtering policy."""

    groups: dict[str, CollisionGroupCfg] = {}
    """Collision groups keyed by their stable policy names."""

    def validate_config(self) -> None:
        """Validate selectors and cross-group references."""
        if not isinstance(self.groups, dict):
            raise TypeError("CollisionFilterCfg.groups must be a dictionary.")
        for group_name, group_cfg in self.groups.items():
            if not isinstance(group_name, str):
                raise TypeError("CollisionFilterCfg.groups keys must be strings.")
            if not isinstance(group_cfg, CollisionGroupCfg):
                raise TypeError(
                    f"Collision group {group_name!r} must be a CollisionGroupCfg, got {type(group_cfg).__name__}."
                )
            group_cfg.validate_config()

        known_groups = set(self.groups)
        unknown_groups = sorted(
            {
                referenced_group
                for group_cfg in self.groups.values()
                for referenced_group in group_cfg.filtered_groups
                if referenced_group not in known_groups
            }
        )
        if unknown_groups:
            raise ValueError(f"Collision filter references unknown groups: {', '.join(unknown_groups)}.")


@configclass
class PhysicsCfg:
    """Abstract base configuration for physics managers.

    This base class contains parameters shared by every physics backend.
    Subclasses should override the class_type to return the appropriate
    physics manager class.

    Shared simulation parameters (dt, gravity, physics_prim_path, physics_material)
    are read directly from :class:`SimulationCfg` by the physics manager.
    """

    class_type: type[PhysicsManager] | Any = MISSING
    """The physics manager class to use. Must be set by subclasses."""

    deterministic: bool = False
    """Whether to request reproducible physics from the backend. Defaults to False.

    This is the backend-agnostic form of the request, set by the ``--deterministic`` command-line
    flag. Each physics manager translates it into its own settings when the simulation starts, and
    raises when its configuration cannot provide the guarantee. A backend-specific determinism
    attribute set explicitly, such as
    :attr:`~isaaclab_newton.physics.NewtonCfg.deterministic_mode`, is the more specific instruction
    and takes precedence.

    Deterministic execution can increase memory use and reduce simulation performance.
    """

    collision_filter: CollisionFilterCfg | None = None
    """Declarative collider policy applied at the clone-plan assembly barrier. Defaults to ``None``.

    Environment isolation is cloning policy and is intentionally not represented here. The active
    physics manager realizes both inputs at the collision-filter application barrier. A workflow
    using this policy must call :func:`isaaclab.cloner.replicate`, even for one environment; model
    construction rejects a configured policy that never crossed the barrier.
    """

    def validate_config(self) -> None:
        """Validate backend-neutral physics configuration."""
        if self.collision_filter is not None:
            if not isinstance(self.collision_filter, CollisionFilterCfg):
                raise TypeError(
                    "PhysicsCfg.collision_filter must be a CollisionFilterCfg or None, got "
                    f"{type(self.collision_filter).__name__}."
                )
            self.collision_filter.validate_config()


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
        from isaaclab_ov.physics import OvPhysxCfg

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
    assert selected is not None
    return selected.replace(
        deterministic=physics_cfg.deterministic or selected.deterministic,
        collision_filter=(
            physics_cfg.collision_filter if physics_cfg.collision_filter is not None else selected.collision_filter
        ),
    )
