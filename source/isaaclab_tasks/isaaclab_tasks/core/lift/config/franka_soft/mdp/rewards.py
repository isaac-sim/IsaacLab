# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the deformable lift tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def deformable_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward if the deformable COM is above a minimum height.

    Args:
        env: The environment instance.
        minimal_height: Minimum COM height [m].
        asset_cfg: The deformable object entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w.torch[:, 2]
    return torch.where(com_z > minimal_height, 1.0, 0.0)


def deformable_lifting(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Reward raising the deformable COM above ``minimal_height`` [m] using a tanh kernel with scale ``std`` [m].

    Dense analogue of :func:`deformable_lifted`: ungated and continuous, so it supplies a smooth
    upward gradient rather than a binary step. Returns ``0`` at or below ``minimal_height`` and
    saturates toward ``1``.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        minimal_height: Minimum COM height [m].
        asset_cfg: The deformable object entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w.torch[:, 2]
    height = (com_z - minimal_height).clamp(min=0.0)
    return torch.tanh(height / std)


def deformable_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward reaching the deformable's nearest nodal point with the end-effector.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    nodal_pos_w = asset.data.nodal_pos_w.torch
    ee_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.linalg.norm(nodal_pos_w - ee_w.unsqueeze(1), dim=2).min(dim=1).values
    return 1.0 - torch.tanh(distance / std)


def deformable_com_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward reaching the deformable's center of mass with the end-effector using a tanh kernel.

    Uses the COM (:attr:`~isaaclab.assets.DeformableObject.data.root_pos_w`) rather than the nearest
    node, so the gripper is drawn to the object's middle. For an elongated body (e.g. a beam) this
    steers the grasp toward the center instead of an end, keeping the object balanced when lifted.

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        asset_cfg: The deformable object entity.
        ee_frame_cfg: The end-effector frame entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    com_w = asset.data.root_pos_w.torch
    ee_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance = torch.linalg.norm(com_w - ee_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def deformable_fingertip_distance(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_com: bool = False,
) -> torch.Tensor:
    """Reward closing the gripper around the deformable using a tanh kernel with scale ``std`` [m].

    Each selected finger body is rewarded for approaching the nearest deformable node
    (:attr:`~isaaclab.assets.DeformableObject.data.nodal_pos_w`),
    so grasping any part of the soft body is credited rather than only its center. Supplies the grasp
    gradient that the EE-reach reward lacks. When ``target_com`` is set, each finger is instead drawn
    to the object's center of mass, biasing the grasp to the middle of an elongated body (e.g. a beam).

    Args:
        env: The environment instance.
        std: The tanh kernel standard deviation [m].
        asset_cfg: The deformable object entity.
        robot_cfg: The robot entity with ``body_ids`` selecting the finger bodies.
        target_com: If ``True``, target the COM instead of the nearest node.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    # target points in world frame: COM (num_envs, 1, 3) or all nodes (num_envs, num_nodes, 3)
    if target_com:
        target_w = asset.data.root_pos_w.torch.unsqueeze(1)
    else:
        target_w = asset.data.nodal_pos_w.torch
    # selected finger bodies in world frame: (num_envs, num_fingers, 3)
    finger_pos_w = robot.data.body_pos_w.torch[:, robot_cfg.body_ids]
    # nearest target to each finger: (num_envs, num_fingers)
    distance = torch.linalg.norm(finger_pos_w.unsqueeze(2) - target_w.unsqueeze(1), dim=3)
    nearest = distance.min(dim=2).values
    return (1.0 - torch.tanh(nearest / std)).mean(dim=1)


class deformable_com_goal_distance(ManagerTermBase):
    """Reward tracking of the goal position by the deformable's COM (tanh kernel).

    Only credits when the COM is above ``minimal_height`` [m] (i.e. the object is lifted).
    The command is interpreted as ``[x, y, z, qw, qx, qy, qz]`` in the robot's root frame.

    If ``success_threshold`` is provided in the term params, this also tracks per-episode
    success (sticky binary: COM ever within ``success_threshold`` [m] of the commanded goal
    while lifted above ``minimal_height``) and logs the mean across environments under
    ``Metrics/success_rate`` on reset.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._track_success = cfg.params.get("success_threshold") is not None
        if self._track_success:
            self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        if self._track_success:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self._succeeded[env_ids].float().mean().item()
            )
            self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        minimal_height: float,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
        success_threshold: float | None = None,
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        asset: DeformableObject = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
        )
        com_w = asset.data.root_pos_w.torch
        distance = torch.linalg.norm(des_pos_w - com_w, dim=1)
        is_lifted = com_w[:, 2] > minimal_height
        if success_threshold is not None:
            self._succeeded |= is_lifted & (distance < success_threshold)
        return is_lifted.float() * (1.0 - torch.tanh(distance / std))


def deformable_com_goal_reached(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    command_name: str,
    success_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Per-step success bonus for holding the deformable COM at the goal.

    Returns ``1.0`` while the COM is within ``success_threshold`` [m] of the commanded goal
    position and lifted above ``minimal_height`` [m], else ``0.0``. Matches the condition
    tracked as ``Metrics/success_rate`` in :class:`deformable_com_goal_distance`.

    Args:
        env: The environment instance.
        minimal_height: Minimum COM height for the bonus to apply [m].
        command_name: Name of the goal-pose command term.
        success_threshold: Maximum COM-to-goal distance counted as success [m].
        robot_cfg: The robot entity providing the goal reference frame.
        asset_cfg: The deformable object entity.

    Returns:
        Reward tensor with shape ``(num_envs,)`` valued in ``{0, 1}``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    asset: DeformableObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    com_w = asset.data.root_pos_w.torch
    distance = torch.linalg.norm(des_pos_w - com_w, dim=1)
    is_lifted = com_w[:, 2] > minimal_height
    return (is_lifted & (distance < success_threshold)).float()
