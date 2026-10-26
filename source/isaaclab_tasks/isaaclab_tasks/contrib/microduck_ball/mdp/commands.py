# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity command of the MicroDuck on a ball."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import BallVelocityCommandCfg


class BallVelocityCommand(UniformVelocityCommand):
    """Velocity command given in a reference heading frame, with a stepping clock.

    A velocity command expressed in the frame of the robot can be met by walking in circles on top of a ball that does
    not move. The sampled command is therefore a direction in a *reference heading* frame: it starts at the heading of
    the robot when an episode begins and then only turns with the commanded yaw rate. The robot observes the target
    velocity in its own heading frame, so it sees when it has to correct its heading.

    The command also owns a stepping clock whose frequency grows with the commanded speed. It stands still while the
    command asks for no motion.
    """

    cfg: BallVelocityCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: BallVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.reference_heading = torch.zeros(self.num_envs, device=self.device)
        """Reference heading in the world frame [rad], shape (num_envs,)."""
        self.gait_phase = torch.rand(self.num_envs, device=self.device)
        """Phase of the stepping clock in [0, 1), shape (num_envs,)."""
        self._foot_phase_offset = torch.tensor([0.0, 0.5], device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Target velocity (x, y) in the robot heading frame [m/s] and yaw rate [rad/s], shape (num_envs, 3)."""
        return torch.cat([self.target_velocity_b, self.vel_command_b[:, 2:]], dim=-1)

    @property
    def is_moving(self) -> torch.Tensor:
        """Whether the command asks for a speed or a turn, shape (num_envs,)."""
        speed = torch.linalg.norm(self.vel_command_b[:, :2], dim=-1)
        return (speed > self.cfg.moving_speed) | (self.vel_command_b[:, 2].abs() > self.cfg.moving_yaw_rate)

    @property
    def target_velocity_w(self) -> torch.Tensor:
        """Target horizontal velocity in the world frame [m/s], shape (num_envs, 2)."""
        return _rotate(self.vel_command_b[:, :2], self.reference_heading)

    @property
    def target_velocity_b(self) -> torch.Tensor:
        """Target horizontal velocity in the robot heading frame [m/s], shape (num_envs, 2)."""
        return _rotate(self.target_velocity_w, -self.robot.data.heading_w.torch)

    @property
    def foot_phase(self) -> torch.Tensor:
        """Phase of each foot's cycle in [0, 1), the second foot half a cycle behind, shape (num_envs, 2)."""
        return torch.remainder(self.gait_phase.unsqueeze(-1) + self._foot_phase_offset, 1.0)

    @property
    def gait_clock(self) -> torch.Tensor:
        """Sine and cosine of the clock phase, zero while the command asks for no motion, shape (num_envs, 2)."""
        angle = 2 * torch.pi * self.gait_phase
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1) * self.is_moving.unsqueeze(-1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        extras = super().reset(env_ids)
        env_ids = slice(None) if env_ids is None else env_ids
        self.reference_heading[env_ids] = self.robot.data.heading_w.torch[env_ids]
        self.gait_phase[env_ids] = torch.rand_like(self.gait_phase[env_ids])
        return extras

    def _update_metrics(self):
        # the target is a world velocity, so the tracking error is measured on the world velocity of the robot
        error_xy = torch.linalg.norm(self.target_velocity_w - self.robot.data.root_lin_vel_w.torch[:, :2], dim=-1)
        error_yaw = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b.torch[:, 2])
        self._error_xy_sum += error_xy
        self._error_yaw_sum += error_yaw
        self._step_count += 1.0

    def _update_command(self):
        super()._update_command()
        self.reference_heading += self.vel_command_b[:, 2] * self._env.step_dt
        speed = torch.linalg.norm(self.vel_command_b[:, :2], dim=-1)
        frequency = (
            self.cfg.base_frequency
            + self.cfg.speed_frequency * speed
            + self.cfg.turn_frequency * self.vel_command_b[:, 2].abs()
        )
        self.gait_phase = torch.remainder(self.gait_phase + frequency * self._env.step_dt * self.is_moving, 1.0)


def _rotate(vector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate planar vectors of shape (N, 2) by angles of shape (N,) [rad]."""
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack([cos * vector[:, 0] - sin * vector[:, 1], sin * vector[:, 0] + cos * vector[:, 1]], dim=-1)
