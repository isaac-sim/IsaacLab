# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-walking commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
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
        r = torch.empty(num_envs, device=self.device)
        for dim, (low, high) in enumerate(self.cfg.ranges):
            self._command[env_ids, dim] = r.uniform_(low, high)


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
    """Probability that an environment is commanded to walk straight forward. Defaults to 0.0."""
    rel_turn_in_place_envs: float = 0.0
    """Probability that an environment is commanded to turn on the spot. Defaults to 0.0."""
    forward_min_speed: float = 0.3
    """Minimum forward speed of forward-only commands [m/s]."""
    turn_in_place_min_fraction: float = 0.4
    """Minimum turn-in-place yaw rate, as a fraction of the largest ``ang_vel_z`` bound [-]."""
