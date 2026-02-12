# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the dexsuite task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class out_of_bound(ManagerTermBase):
    """Termination condition for when the object falls out of bound.

    This class-based implementation caches the asset reference, ranges tensor, and env_origins
    to avoid recomputing them on every call.
    """

    def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRLEnv"):
        """Initialize the termination term.

        Args:
            cfg: The termination term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # Cache asset reference
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._object: RigidObject = env.scene[asset_cfg.name]

        # Cache ranges tensor
        in_bound_range: dict[str, tuple[float, float]] = cfg.params.get("in_bound_range", {})
        range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        self._ranges = torch.tensor(range_list, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        in_bound_range: dict[str, tuple[float, float]] = {},
    ) -> torch.Tensor:
        """Check if the object is out of bounds.

        Args:
            env: The environment (unused, cached in __init__).
            asset_cfg: The object configuration (unused, cached in __init__).
            in_bound_range: The bound ranges (unused, cached in __init__).

        Returns:
            Boolean tensor indicating which environments have objects out of bounds.
        """
        object_pos_local = wp.to_torch(self._object.data.root_pos_w) - env.scene.env_origins
        outside_bounds = ((object_pos_local < self._ranges[:, 0]) | (object_pos_local > self._ranges[:, 1])).any(dim=1)
        return outside_bounds


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    return (wp.to_torch(robot.data.joint_vel).abs() > (wp.to_torch(robot.data.joint_vel_limits) * 2)).any(dim=1)
