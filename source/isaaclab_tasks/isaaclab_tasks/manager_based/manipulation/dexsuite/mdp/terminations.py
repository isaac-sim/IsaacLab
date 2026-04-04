# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the dexsuite task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


class out_of_bound(ManagerTermBase):
    """Termination condition for when the object falls out of bound.

    This class-based implementation caches the asset reference, ranges tensor, and env_origins
    to avoid recomputing them on every call.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._object: RigidObject = env.scene[asset_cfg.name]

        in_bound_range: dict[str, tuple[float, float]] = cfg.params.get("in_bound_range", {})
        range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=env.device, dtype=torch.float32)

        # Pre-apply env_origins so we can compare directly against world-space positions.
        origins = env.scene.env_origins  # (N, 3)
        self._lower = origins + ranges[:, 0]  # (N, 3)
        self._upper = origins + ranges[:, 1]  # (N, 3)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        in_bound_range: dict[str, tuple[float, float]] = {},
    ) -> torch.Tensor:
        pos_w = wp.to_torch(self._object.data.root_pos_w)
        return ((pos_w < self._lower) | (pos_w > self._upper)).any(dim=1)


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = wp.to_torch(robot.data.joint_vel)
    joint_vel_limits = wp.to_torch(robot.data.joint_vel_limits)
    return (joint_vel.abs() > (joint_vel_limits * 2)).any(dim=1)
