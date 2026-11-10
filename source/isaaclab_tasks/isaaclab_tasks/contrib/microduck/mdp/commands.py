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
from isaaclab.utils.warp import ProxyArray

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

    def _resample_command(self, env_mask: ProxyArray):
        # Read the bounds on every resample: the curriculum widens the head-pose ranges during training.
        low, high = torch.tensor(self.cfg.ranges, device=self.device).unbind(-1)
        sample = low + (high - low) * torch.rand(self.num_envs, self.dim, device=self.device)
        torch.where(env_mask.torch[:, None], sample, self._command, out=self._command)


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

    def _resample_command(self, env_mask: ProxyArray):
        super()._resample_command(env_mask)
        mask, r, vel = env_mask.torch, self._sample, self.vel_command_b
        forward = mask & (r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs)
        vel[:, 0] = torch.where(forward, vel[:, 0].abs().clamp(min=self.cfg.forward_min_speed), vel[:, 0])
        vel[:, 1:].masked_fill_(forward[:, None], 0.0)
        if self.cfg.rel_turn_in_place_envs <= 0.0:
            return
        turn = mask & (r.uniform_(0.0, 1.0) < self.cfg.rel_turn_in_place_envs)
        vel[:, :2].masked_fill_(turn[:, None], 0.0)
        low, high = self.cfg.ranges.ang_vel_z
        max_rate = max(abs(low), abs(high))
        sign = torch.where(r.uniform_(0.0, 1.0) < 0.5, -1.0, 1.0)
        magnitude = r.uniform_(self.cfg.turn_in_place_min_fraction * max_rate, max_rate)
        vel[:, 2] = torch.where(turn, sign * magnitude, vel[:, 2])
        self.is_standing_env.masked_fill_(turn, False)


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

    def reset(self, env_mask: ProxyArray) -> dict[str, float]:
        """Clear progress after the curriculum has consumed the completed episodes."""
        extras = super().reset(env_mask)
        for value in self.terrain_progress.values():
            value.masked_fill_(env_mask.torch, 0.0)
        self._last_progress_step.masked_fill_(env_mask.torch, 0)
        return extras
