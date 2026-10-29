# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A downward suction tool controlled by Cartesian residuals and a binary vacuum."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from scipy.optimize import least_squares

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTerm

from ..conveyor_franka.mdp.reset_events import franka_tool_position
from .actions_cfg import DOWN_QUATERNION, CartesianSuctionArmCfg, VacuumActionCfg


class CartesianSuctionArm(DifferentialInverseKinematicsAction):
    """Reuse the standard six-dimensional IK solver with a fixed downward orientation."""

    def __init__(self, cfg: CartesianSuctionArmCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.invalid_actions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.previous_actions = torch.zeros_like(self._raw_actions)
        self._needs_reference = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._down = torch.tensor(DOWN_QUATERNION, device=self.device).expand(self.num_envs, -1)
        self._lower = torch.tensor((-0.45, -0.95, 0.045), device=self.device)
        self._upper = torch.tensor((1.0, 0.95, 0.75), device=self.device)
        self._shoulder = torch.tensor((0.0, 0.0, 0.333), device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    def process_actions(self, actions: torch.Tensor) -> None:
        self.invalid_actions[:] = ~torch.isfinite(actions).all(dim=-1)
        self.previous_actions.copy_(self._raw_actions)
        self._raw_actions[:] = torch.nan_to_num(actions).clamp(-1.0, 1.0)
        self._processed_actions[:] = self._raw_actions * self._scale
        position, rotation = self._compute_frame_pose()
        reference = torch.where(self._needs_reference[:, None], position, self._ik_controller.ee_pos_des)
        desired = (reference + self._processed_actions).clamp(min=self._lower, max=self._upper)
        displacement = desired - self._shoulder
        radius = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True).clamp_min(1.0e-6)
        desired = self._shoulder + displacement * (0.78 / radius).clamp(max=1.0)
        self._ik_controller.set_command(torch.cat((desired, self._down), dim=-1), position, rotation)
        self._needs_reference[:] = False

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.invalid_actions[slice(None) if env_ids is None else env_ids] = False
        self.previous_actions[slice(None) if env_ids is None else env_ids] = 0.0
        self._needs_reference[slice(None) if env_ids is None else env_ids] = True


class VacuumAction(ActionTerm):
    """Signed binary command: negative seals, nonnegative releases."""

    def __init__(self, cfg: VacuumActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.previous_actions = torch.zeros_like(self._raw_actions)
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
        self.previous_actions.copy_(self._raw_actions)
        self._raw_actions[:] = torch.nan_to_num(actions).clamp(-1.0, 1.0)
        self._processed_actions[:] = torch.where(self._raw_actions < 0.0, -1.0, 1.0)

    def apply_actions(self) -> None:
        self._env.gripper.set_command(self._processed_actions[:, 0] < 0.0, self._env.active_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 1.0
        self.invalid_actions[ids] = False
        self.previous_actions[ids] = 0.0


def solve_tool_ik(positions: torch.Tensor) -> torch.Tensor:
    """Generate collision-screenable downward Panda reset poses at startup, on the CPU."""
    targets = positions.detach().cpu().numpy()
    solutions = []
    for target in targets:
        angle = math.atan2(float(target[1]), float(target[0]))

        def joints(values: np.ndarray) -> torch.Tensor:
            shoulder, elbow = values
            return torch.tensor((angle, shoulder, 0.0, elbow, 0.0, shoulder - elbow, angle), dtype=torch.float64)

        def error(values: np.ndarray) -> np.ndarray:
            current = franka_tool_position(joints(values).unsqueeze(0), tool_offset_z=0.212)[0]
            return current.numpy() - target

        fit = least_squares(error, (0.35, -2.0), bounds=((-1.7, -3.0), (1.7, -0.15)), max_nfev=80)
        if np.linalg.norm(fit.fun) > 0.003:
            raise ValueError(f"Unreachable suction reset position {target.tolist()}: {fit.fun.tolist()}")
        solutions.append(joints(fit.x).to(dtype=positions.dtype))
    return torch.stack(solutions).to(device=positions.device)
