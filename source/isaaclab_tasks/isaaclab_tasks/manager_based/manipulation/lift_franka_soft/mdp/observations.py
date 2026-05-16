# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the Franka deformable / cable lifting environments.

Both volumetric/surface deformables and cables expose a per-asset point cloud
(``nodal_pos_w`` or ``body_pos_w``); :func:`_points_w` and :func:`_com_w` (in
:mod:`.rewards`) dispatch to the right attribute so downstream observation math
is shared between the two task families.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .rewards import _com_w, _points_w

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def object_com_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the asset's COM in the robot's root frame [m].

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    asset = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    com_w = _com_w(asset)
    com_b, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, com_w)
    return com_b


class ObjectSampledPointsInRobotRootFrame(ManagerTermBase):
    """Sampled asset point positions expressed in the robot's root frame.

    Works for both volumetric/surface deformables (sampling nodal points) and
    cable articulations (sampling per-segment frames). The point indices are
    sampled on reset and reused within the episode so each observed point
    follows the same material node / segment over time.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.num_points: int = cfg.params.get("num_points", 20)

        asset = env.scene[self.asset_cfg.name]
        self.num_source_points = _points_w(asset).shape[1]
        self.point_ids = torch.empty(env.num_envs, self.num_points, dtype=torch.long, device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample observed point indices for the selected environments."""
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        if self.num_points <= self.num_source_points:
            self.point_ids[env_ids] = (
                torch.rand((num_envs, self.num_source_points), device=self.device).topk(self.num_points, dim=1).indices
            )
        else:
            self.point_ids[env_ids] = torch.randint(
                self.num_source_points, (num_envs, self.num_points), device=self.device
            )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        num_points: int = 20,
    ) -> torch.Tensor:
        """Sample asset point positions in the robot's root frame.

        Args:
            env: The environment instance.
            asset_cfg: The deformable or cable entity.
            robot_cfg: The robot entity providing the reference frame.
            num_points: Number of sampled points.

        Returns:
            Flattened tensor of shape ``(num_envs, 3 * num_points)`` with sampled
            point positions [m] in the robot root frame.
        """
        asset = env.scene[asset_cfg.name]
        robot: Articulation = env.scene[robot_cfg.name]
        if num_points != self.num_points:
            raise ValueError(f"Requested {num_points} points, but this term was initialized with {self.num_points}.")

        points_w = _points_w(asset)
        sampled_points_w = points_w.gather(1, self.point_ids.unsqueeze(-1).expand(-1, -1, 3))

        flat_sampled_points_w = sampled_points_w.reshape(-1, 3)
        root_pos_w = robot.data.root_pos_w.torch.unsqueeze(1).expand(-1, num_points, -1)
        root_quat_w = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, num_points, -1)
        sampled_points_b, _ = subtract_frame_transforms(
            root_pos_w.reshape(-1, 3),
            root_quat_w.reshape(-1, 4),
            flat_sampled_points_w,
        )
        return sampled_points_b.view(env.num_envs, -1)
