# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A downward suction tool controlled by Cartesian residuals and a binary vacuum."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTerm

from ..conveyor_franka.mdp.reset_events import franka_tool_position
from .actions_cfg import (
    DOWN_QUATERNION,
    LINK7_TO_TOOL_Z,
    ROBOT_SHOULDER_Z,
    TOOL_REACH_RADIUS,
    CartesianSuctionArmCfg,
    VacuumActionCfg,
)


class CartesianSuctionArm(DifferentialInverseKinematicsAction):
    """Reuse the standard six-dimensional IK solver with a fixed downward orientation."""

    def __init__(self, cfg: CartesianSuctionArmCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.invalid_actions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._needs_reference = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._down = torch.tensor(DOWN_QUATERNION, device=self.device).expand(self.num_envs, -1)
        self._lower = torch.tensor((-0.68, -0.95, 0.045), device=self.device)
        self._upper = torch.tensor((1.0, 0.95, 0.75), device=self.device)
        self._shoulder = torch.tensor((0.0, 0.0, ROBOT_SHOULDER_Z), device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    def process_actions(self, actions: torch.Tensor) -> None:
        self.invalid_actions[:] = ~torch.isfinite(actions).all(dim=-1)
        self._raw_actions[:] = torch.nan_to_num(actions).clamp(-1.0, 1.0)
        self._processed_actions[:] = self._raw_actions * self._scale
        position, rotation = self._compute_frame_pose()
        reference = torch.where(self._needs_reference[:, None], position, self._ik_controller.ee_pos_des)
        desired = (reference + self._processed_actions).clamp(min=self._lower, max=self._upper)
        displacement = desired - self._shoulder
        radius = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True).clamp_min(1.0e-6)
        desired = self._shoulder + displacement * (TOOL_REACH_RADIUS / radius).clamp(max=1.0)
        lead = desired - position
        distance = torch.linalg.vector_norm(lead, dim=-1, keepdim=True).clamp_min(1.0e-6)
        desired = position + lead * (self.cfg.max_target_error / distance).clamp(max=1.0)
        self._ik_controller.set_command(torch.cat((desired, self._down), dim=-1), position, rotation)
        self._needs_reference[:] = False

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.invalid_actions[slice(None) if env_ids is None else env_ids] = False
        self._needs_reference[slice(None) if env_ids is None else env_ids] = True


class VacuumAction(ActionTerm):
    """Signed binary command: negative seals, nonnegative releases."""

    def __init__(self, cfg: VacuumActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.invalid_actions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self.invalid_actions[:] = ~torch.isfinite(actions).all(dim=-1)
        self._raw_actions[:] = torch.nan_to_num(actions).clamp(-1.0, 1.0)
        self._processed_actions[:] = torch.where(self._raw_actions < 0.0, -1.0, 1.0)
        self._env.command_manager.get_term("sort").record_release(self._processed_actions[:, 0] > 0.0)

    def apply_actions(self) -> None:
        self._env.gripper.set_command(self._processed_actions[:, 0] < 0.0, self._env.active_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 1.0
        self.invalid_actions[ids] = False


def solve_tool_ik(positions: torch.Tensor) -> torch.Tensor:
    """Solve the downward Panda reset family in a batch, rejecting unreachable poses."""
    if (
        positions.ndim != 2
        or positions.shape[-1] != 3
        or not positions.is_floating_point()
        or not torch.isfinite(positions).all()
    ):
        raise ValueError("Suction reset positions must be a finite floating-point tensor of shape (N, 3).")
    # With joints 3/5 fixed and joint 6 cancelling the elbow, Panda FK is a planar two-link arm.
    first_length, second_length = math.hypot(0.316, 0.0825), math.hypot(0.384, 0.0825)
    first_offset, second_offset = math.atan2(0.0825, 0.316), math.atan2(0.0825, 0.384)
    radial = positions[:, :2].norm(dim=-1) - 0.088
    height = positions[:, 2] - ROBOT_SHOULDER_Z + LINK7_TO_TOOL_Z
    cosine = (radial.square() + height.square() - first_length**2 - second_length**2) / (
        2.0 * first_length * second_length
    )
    if (cosine.abs() > 1.0 + 8.0 * torch.finfo(positions.dtype).eps).any():
        raise ValueError("Suction reset position is outside the downward Panda workspace.")
    bend = cosine.clamp(-1.0, 1.0).acos()
    shoulder = (
        torch.atan2(radial, height)
        - torch.atan2(second_length * bend.sin(), first_length + second_length * bend.cos())
        - first_offset
    )
    elbow = -(bend + first_offset + second_offset)
    angle = torch.atan2(positions[:, 1], positions[:, 0])
    zeros = torch.zeros_like(angle)
    joints = torch.stack((angle, shoulder, zeros, elbow, zeros, shoulder - elbow, angle), dim=-1)
    lower = positions.new_tensor((-2.897, -1.7, -2.897, -3.0, -2.897, -0.0175, -2.897))
    upper = positions.new_tensor((2.897, 1.7, 2.897, -0.15, 2.897, 3.7525, 2.897))
    residual = (franka_tool_position(joints, tool_offset_z=LINK7_TO_TOOL_Z) - positions).norm(dim=-1)
    invalid = ((joints < lower) | (joints > upper)).any(dim=-1) | (residual > 0.003)
    if invalid.any():
        raise ValueError(f"Unreachable suction reset position {positions[invalid][0].tolist()}.")
    return joints
