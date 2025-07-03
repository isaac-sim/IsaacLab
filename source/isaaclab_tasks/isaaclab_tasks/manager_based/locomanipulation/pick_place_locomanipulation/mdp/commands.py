# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands import UniformVelocityCommand

# Only import the command class during type checking
if TYPE_CHECKING:
    from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation.configs.locomanipulation_commands_cfg import (
        UniformVelocityAndHeightCommandCfg,
    )

    # from ..configs.locomanipulation_commands_cfg import UniformVelocityAndHeightCommandCfg

# Hard code this parameter for now. Need to have a better design here.
NORMAL_WALK_HEIGHT = 0.70


class UniformVelocityAndHeightCommand(UniformVelocityCommand):
    """Uniform velocity command generator with height command."""

    cfg: UniformVelocityAndHeightCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityAndHeightCommandCfg, env):
        """Initialize the command generator.

        Args:
            cfg: Command configuration.
            env: Environment instance.
        """
        super().__init__(cfg, env)
        # Height command
        self._height_command = torch.zeros(self.num_envs, 1, device=self.device)
        # -- metrics
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommandWithHeight:\n"
        msg += f"\tCommand dimension: {tuple(self.commands.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
            msg += f"\tHeading command: {self.cfg.heading_command}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def height_command(self) -> torch.Tensor:
        """The desired height command. Shape is (num_envs, 1)."""
        return self._height_command

    @property
    def vel_command(self) -> torch.Tensor:
        """The desired velocity command. Shape is (num_envs, 3)."""
        return self.vel_command_b

    @property
    def command(self) -> torch.Tensor:
        """The desired command. Shape is (num_envs, 4)."""
        return torch.cat([self.vel_command_b, self.height_command], dim=-1)

    @property
    def is_standing_flag(self) -> torch.Tensor:
        """The flag indicating if the robot is standing.

        Returns:
            is_standing_flag (num_envs, 1): The flag indicating if the robot is standing.
        """
        return self.is_standing_env.unsqueeze(-1).float()

    def _update_metrics(self):
        super()._update_metrics()
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data - squeeze height_command to match root_pos_w shape
        self.metrics["error_height"] += (
            torch.abs(self._height_command.squeeze(-1) - self.robot.data.root_pos_w[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands and heading command.
        super()._resample_command(env_ids)
        # Sample height command
        r = torch.empty(len(env_ids), device=self.device)
        self._height_command[env_ids] = r.uniform_(*self.cfg.ranges.height).unsqueeze(-1)

    def _update_command(self):
        """Post-processes the velocity command.

        This function 1) sets velocity command to zero for standing environments, 2) computes
        angular velocity from heading direction if the heading_command flag is set and 3) sets
        height command a constant height for non-standing environments.
        """
        super()._update_command()

        non_standing_env_ids = (~self.is_standing_env).nonzero(as_tuple=False).flatten()
        self._height_command[non_standing_env_ids] = NORMAL_WALK_HEIGHT
