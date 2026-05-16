# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the Franka deformable lifting environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv


def deformable_com_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the deformable object's COM in the robot's root frame [m].

    The COM is the mean of the deformable's nodal positions (see
    :attr:`~isaaclab.assets.DeformableObject.data.root_pos_w`).

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    com_w = wp.to_torch(asset.data.root_pos_w)
    com_b, _ = subtract_frame_transforms(wp.to_torch(robot.data.root_pos_w), wp.to_torch(robot.data.root_quat_w), com_w)
    return com_b


class DeformableSampledPointsInRobotRootFrame(ManagerTermBase):
    """Sampled deformable nodal points expressed in the robot's root frame.

    The point indices are sampled on reset, then reused within the episode so
    each observed point follows the same material node over time.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("deformable"))
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.num_points: int = cfg.params.get("num_points", 20)

        asset: DeformableObject = env.scene[self.asset_cfg.name]
        self.num_nodes = asset.data.nodal_pos_w.shape[1]
        self.node_ids = torch.empty(env.num_envs, self.num_points, dtype=torch.long, device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample observed deformable nodes for the selected environments."""
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        if self.num_points <= self.num_nodes:
            self.node_ids[env_ids] = (
                torch.rand((num_envs, self.num_nodes), device=self.device).topk(self.num_points, dim=1).indices
            )
        else:
            self.node_ids[env_ids] = torch.randint(self.num_nodes, (num_envs, self.num_points), device=self.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        num_points: int = 20,
    ) -> torch.Tensor:
        """Sample deformable nodal positions in the robot's root frame.

        Args:
            env: The environment instance.
            asset_cfg: The deformable object entity.
            robot_cfg: The robot entity providing the reference frame.
            num_points: Number of sampled points.

        Returns:
            Flattened tensor of shape ``(num_envs, 3 * num_points)`` with sampled
            point positions [m] in the robot root frame.
        """
        asset: DeformableObject = env.scene[asset_cfg.name]
        robot: Articulation = env.scene[robot_cfg.name]
        if num_points != self.num_points:
            raise ValueError(
                f"Requested {num_points} deformable points, but this term was initialized with {self.num_points}."
            )

        nodal_pos_w = asset.data.nodal_pos_w.torch
        sampled_points_w = nodal_pos_w.gather(1, self.node_ids.unsqueeze(-1).expand(-1, -1, 3))

        flat_sampled_points_w = sampled_points_w.reshape(-1, 3)
        root_pos_w = robot.data.root_pos_w.torch.unsqueeze(1).expand(-1, num_points, -1)
        root_quat_w = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, num_points, -1)
        sampled_points_b, _ = subtract_frame_transforms(
            root_pos_w.reshape(-1, 3),
            root_quat_w.reshape(-1, 4),
            flat_sampled_points_w,
        )
        return sampled_points_b.view(env.num_envs, -1)


class CableSampledPointsInRobotRootFrame(ManagerTermBase):
    """Sampled cable segment positions expressed in the robot's root frame.

    A cable is a Newton articulation whose bodies are the per-segment rigid
    frames produced by :meth:`newton.ModelBuilder.add_rod_graph`; the segment
    indices are sampled on reset and reused within the episode so each observed
    point follows the same segment over time.

    Mirrors :class:`DeformableSampledPointsInRobotRootFrame` but accesses
    :attr:`~isaaclab.assets.Articulation.data.body_pos_w` instead of
    :attr:`~isaaclab.assets.DeformableObject.data.nodal_pos_w`.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("cable"))
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.num_points: int = cfg.params.get("num_points", 20)

        asset: Articulation = env.scene[self.asset_cfg.name]
        self.num_segments = asset.data.body_pos_w.shape[1]
        self.segment_ids = torch.empty(env.num_envs, self.num_points, dtype=torch.long, device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resample observed cable segments for the selected environments."""
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        if self.num_points <= self.num_segments:
            self.segment_ids[env_ids] = (
                torch.rand((num_envs, self.num_segments), device=self.device).topk(self.num_points, dim=1).indices
            )
        else:
            self.segment_ids[env_ids] = torch.randint(
                self.num_segments, (num_envs, self.num_points), device=self.device
            )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        num_points: int = 20,
    ) -> torch.Tensor:
        """Sample cable segment positions in the robot's root frame.

        Args:
            env: The environment instance.
            asset_cfg: The cable articulation entity.
            robot_cfg: The robot entity providing the reference frame.
            num_points: Number of sampled points.

        Returns:
            Flattened tensor of shape ``(num_envs, 3 * num_points)`` with sampled
            segment positions [m] in the robot root frame.
        """
        asset: Articulation = env.scene[asset_cfg.name]
        robot: Articulation = env.scene[robot_cfg.name]
        if num_points != self.num_points:
            raise ValueError(
                f"Requested {num_points} cable points, but this term was initialized with {self.num_points}."
            )

        body_pos_w = asset.data.body_pos_w.torch
        sampled_points_w = body_pos_w.gather(1, self.segment_ids.unsqueeze(-1).expand(-1, -1, 3))

        flat_sampled_points_w = sampled_points_w.reshape(-1, 3)
        root_pos_w = robot.data.root_pos_w.torch.unsqueeze(1).expand(-1, num_points, -1)
        root_quat_w = robot.data.root_quat_w.torch.unsqueeze(1).expand(-1, num_points, -1)
        sampled_points_b, _ = subtract_frame_transforms(
            root_pos_w.reshape(-1, 3),
            root_quat_w.reshape(-1, 4),
            flat_sampled_points_w,
        )
        return sampled_points_b.view(env.num_envs, -1)


def cable_com_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the cable midpoint in the robot's root frame [m].

    The midpoint is the mean of the cable's per-segment positions
    (:attr:`~isaaclab.assets.Articulation.data.body_pos_w`).

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    com_w = asset.data.body_pos_w.torch.mean(dim=1)
    com_b, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, com_w)
    return com_b
