# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_experimental.managers.scene_entity_cfg import SceneEntityCfg

import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@wp.func
def wrap_to_pi(angle: float) -> float:
    """Wrap input angle (in radians) to the range [-pi, pi)."""
    two_pi = 2.0 * wp.pi
    wrapped_angle = angle + wp.pi
    # NOTE: Use floor-based remainder semantics to match torch's `%` for negative inputs.
    wrapped_angle = wrapped_angle - wp.floor(wrapped_angle / two_pi) * two_pi
    return wp.where((wrapped_angle == 0) and (angle > 0), wp.pi, wrapped_angle - wp.pi)


def resolve_asset_cfg(cfg: dict, env: ManagerBasedEnv) -> SceneEntityCfg:
    asset_cfg = None

    for value in cfg.values():
        # If it exists, the SceneEntityCfg should have been resolved by the base manager.
        if isinstance(value, SceneEntityCfg):
            asset_cfg = value
            # Check if the joint ids are not set, and if so, resolve them.
            if asset_cfg.joint_ids is None or asset_cfg.joint_ids == slice(None):
                asset_cfg.resolve_for_warp(env.scene)
            if asset_cfg.body_ids is None or asset_cfg.body_ids == slice(None):
                asset_cfg.resolve_for_warp(env.scene)
            break

    # If it doesn't exist, use the default robot entity.
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
        asset_cfg.resolve_for_warp(env.scene)

    return asset_cfg
