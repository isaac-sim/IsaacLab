# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_tasks.contrib.nist.utils.collision_analyzer import CollisionAnalyzer


@configclass
class CollisionAnalyzerCfg(ManagerTermBaseCfg):
    """Configuration for :class:`CollisionAnalyzer`.

    The analyzer samples a point cloud on the surface of the asset's collision
    geometry, then queries signed distances against every obstacle mesh each
    time it is called. An environment is considered collision-free when all
    sampled points are at least :attr:`min_dist` away from every obstacle.
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

    asset_cfg: SceneEntityCfg = MISSING
    """Scene entity whose bodies are sampled for collision checking."""

    obstacle_cfgs: list[SceneEntityCfg] = MISSING
    """Scene entities treated as obstacles to check against."""
