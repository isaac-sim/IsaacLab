# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recipe of the neural robot dynamics model behind ``physics=newton_nerd`` of the Franka reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab_neural import NeRDRecipe, RecordingSimulation
from newton_neural.data import CommandHeld, CommandNearState, CommandSmooth, CommandSwitch, SamplerUniform
from torch import Tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnvCfg

    from .nerd_recipe_cfg import FrankaReachNeRDRecipeCfg


class FrankaReachNeRDRecipe(NeRDRecipe):
    """Records the arm in free space, from anywhere within its joint limits, under four kinds of joint targets."""

    cfg: FrankaReachNeRDRecipeCfg

    def configure(self, env_cfg: ManagerBasedEnvCfg) -> None:
        # The model has no contact input, so the arm is recorded in free space. Random joint targets would drive it
        # into the table.
        env_cfg.scene.table = None

    def start_states(self, simulation: RecordingSimulation) -> SamplerUniform:
        """Start the arm anywhere within its joint limits, moving, with the gripper at its default opening."""
        low, high, arm = self._arm_bounds(simulation)
        default, speed = simulation.default_joint_pos, self.cfg.state_speed
        return SamplerUniform(
            {
                "joint_q": (torch.where(arm, low, default), torch.where(arm, high, default)),
                "joint_qd": (-speed * arm, speed * arm),
            }
        )

    def joint_targets(self, simulation: RecordingSimulation) -> CommandSwitch:
        """Mix four kinds of arm targets. The finger targets stay at zero, as the task leaves them."""
        cfg = self.cfg
        low, high, arm = self._arm_bounds(simulation)
        closed = torch.zeros_like(low)
        return CommandSwitch(
            {
                CommandHeld(cfg.hold_frames): 1.0,  # Large jumps, then settling
                CommandNearState(cfg.explore_reach * arm): 1.0,  # A new target near the arm every frame
                CommandNearState(cfg.refine_reach * arm): 1.0,  # The same, closer to the arm
                CommandSmooth(cfg.drift_timescale): 1.0,  # Smooth motion
            },
            low=torch.where(arm, low - cfg.target_margin, closed),
            high=torch.where(arm, high + cfg.target_margin, closed),
        )

    def _arm_bounds(self, simulation: RecordingSimulation) -> tuple[Tensor, Tensor, Tensor]:
        """Return the joint limits [rad or m] and the mask ``[joint_count]`` of the arm joints."""
        low, high = simulation.state_bounds()["joint_q"]
        return low, high, torch.arange(len(low), device=low.device) < self.cfg.arm_joint_count
