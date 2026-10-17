# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.managers import ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .collision_analyzer import CollisionAnalyzer


@configclass
class CollisionGeometryCfg(SceneEntityCfg):
    """Select an entity's bodies and geometry for reset safety checks."""

    geometry_policy: Literal["collision", "collision_or_visual"] = "collision"
    """Geometry selected independently for each rigid body.

    ``collision`` uses enabled physical colliders. ``collision_or_visual`` uses those colliders when present,
    otherwise visible meshes with default or render purpose. The fallback does not enable physics collisions.
    """


@configclass
class CollisionAnalyzerCfg(ManagerTermBaseCfg):
    """Configuration for :class:`CollisionAnalyzer`.

    The analyzer samples a point cloud on the surface of the asset's collision
    geometry, then queries signed distances against every obstacle mesh each
    time it is called. An environment is considered collision-free when all
    sampled points are at least :attr:`min_dist` away from every obstacle.

    Bare :class:`SceneEntityCfg` entries preserve all authored colliders, including disabled ones.
    Use :class:`CollisionGeometryCfg` to select enabled colliders and optional visual fallback explicitly.
    """

    func: type[CollisionAnalyzer] | str = "{DIR}.collision_analyzer:CollisionAnalyzer"
    """The callable class to instantiate from this configuration."""

    num_points: int = 32
    """Number of surface points sampled per rigid body of the asset."""

    max_dist: float = 0.5
    """Upper clamp for signed-distance queries [m]. Points farther than this
    from any obstacle surface are clamped to this value."""

    min_dist: float = 0.0
    """Minimum clearance threshold [m]. An environment is collision-free only
    when every sampled point has a signed distance >= this value."""

    min_dist_by_obstacle: dict[str, float] = {}
    """Minimum clearance overrides by obstacle scene-entity name [m]."""

    asset_cfg: SceneEntityCfg = MISSING
    """Scene entity whose bodies are sampled. Use :class:`CollisionGeometryCfg` to select visual fallback geometry."""

    obstacle_cfgs: list[SceneEntityCfg] = MISSING
    """Obstacle entities and body selections. :class:`CollisionGeometryCfg` also selects their geometry policy."""
