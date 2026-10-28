# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck walking commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformPoseDeltaCommand(CommandTerm):
    """Generic N-dimensional uniform pose-delta command."""

    cfg: UniformPoseDeltaCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: UniformPoseDeltaCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term."""
        super().__init__(cfg, env)
        self.dim = len(cfg.ranges)
        self._command = torch.zeros(self.num_envs, self.dim, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Pose deltas [m or rad, depending on dimension], shape (num_envs, dim)."""
        return self._command

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        num_envs = len(env_ids)
        if num_envs == 0:
            return
        # Read the bounds on every resample: the curriculum widens the head-pose ranges during training.
        low, high = torch.tensor(self.cfg.ranges, device=self.device).unbind(-1)
        self._command[env_ids] = low + (high - low) * torch.rand(num_envs, self.dim, device=self.device)


@configclass
class UniformPoseDeltaCommandCfg(CommandTermCfg):
    """Configuration for the N-dimensional uniform pose-delta command term."""

    class_type: type[UniformPoseDeltaCommand] = UniformPoseDeltaCommand
    ranges: tuple[tuple[float, float], ...] = ()
    """Per-dimension sampling bounds [m or rad, depending on dimension]. Length sets the command width."""


class MicroDuckVelocityCommand(UniformVelocityCommand):
    """Velocity command with MicroDuck's forward-only and turn-in-place buckets."""

    cfg: MicroDuckVelocityCommandCfg
    """Configuration for the command term."""

    def _resample_command(self, env_ids: Sequence[int] | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        forward_ids = env_ids[r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs]
        if len(forward_ids) > 0:
            self.vel_command_b[forward_ids, 0] = (
                self.vel_command_b[forward_ids, 0].abs().clamp(min=self.cfg.forward_min_speed)
            )
            self.vel_command_b[forward_ids, 1] = 0.0
            self.vel_command_b[forward_ids, 2] = 0.0
        if self.cfg.rel_turn_in_place_envs <= 0.0:
            return
        turn_ids = env_ids[r.uniform_(0.0, 1.0) < self.cfg.rel_turn_in_place_envs]
        if len(turn_ids) == 0:
            return
        self.vel_command_b[turn_ids, :2] = 0.0
        low, high = self.cfg.ranges.ang_vel_z
        max_rate = max(abs(low), abs(high))
        turn_r = torch.empty(len(turn_ids), device=self.device)
        sign = torch.where(turn_r.uniform_(0.0, 1.0) < 0.5, -1.0, 1.0)
        magnitude = turn_r.uniform_(self.cfg.turn_in_place_min_fraction * max_rate, max_rate)
        self.vel_command_b[turn_ids, 2] = sign * magnitude
        self.is_standing_env[turn_ids] = False


@configclass
class MicroDuckVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the MicroDuck velocity command term."""

    class_type: type[MicroDuckVelocityCommand] = MicroDuckVelocityCommand
    rel_forward_envs: float = 0.0
    """Probability that an environment is drawn into the forward-only bucket. Defaults to 0.0.

    The turn-in-place bucket is drawn independently afterwards and takes precedence, so the effective forward-only
    share is ``rel_forward_envs * (1 - rel_turn_in_place_envs)``.
    """
    rel_turn_in_place_envs: float = 0.0
    """Probability that an environment is commanded to turn on the spot. Defaults to 0.0."""
    forward_min_speed: float = 0.3
    """Minimum forward speed of forward-only commands [m/s]."""
    turn_in_place_min_fraction: float = 0.4
    """Minimum turn-in-place yaw rate, as a fraction of the largest ``ang_vel_z`` bound [-]."""


class MicroDuckRoughVelocityCommand(MicroDuckVelocityCommand):
    """Accumulate walking and tracking before command resampling for terrain progression."""

    def __init__(self, cfg: MicroDuckVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.terrain_progress = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("commanded_distance", "walked_distance", "error_distance", "walking_time", "yaw_error", "time")
        }
        self._last_progress_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def record_terrain_progress(self, env_ids: Sequence[int] | slice = slice(None)) -> None:
        """Record each control sample once, including the terminal sample before reset.

        Distances and integrated XY error are in [m], times in [s], and integrated yaw-rate error in [rad].
        Translation commands of at least 0.05 [m/s] contribute to walking statistics.
        """
        steps = self._env.episode_length_buf[env_ids]
        dt = (steps > self._last_progress_step[env_ids]) * self._env.step_dt
        command = self.command[env_ids]
        velocity = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_link_quat_w.torch[env_ids]),
            self.robot.data.root_link_lin_vel_w.torch[env_ids],
        )[:, :2]
        speed = command[:, :2].norm(dim=-1)
        walking_dt = dt * (speed >= 0.05)
        values = {
            "commanded_distance": speed * walking_dt,
            "walked_distance": velocity.norm(dim=-1) * walking_dt,
            "error_distance": (command[:, :2] - velocity).norm(dim=-1) * walking_dt,
            "walking_time": walking_dt,
            "yaw_error": (command[:, 2] - self.robot.data.root_link_ang_vel_w.torch[env_ids, 2]).abs() * dt,
            "time": dt,
        }
        for name, value in values.items():
            self.terrain_progress[name][env_ids] += value
        self._last_progress_step[env_ids] = steps

    def _update_metrics(self):
        super()._update_metrics()
        self.record_terrain_progress()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Clear progress after the curriculum has consumed the completed episodes."""
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for value in self.terrain_progress.values():
            value[ids] = 0.0
        self._last_progress_step[ids] = 0
        return extras
