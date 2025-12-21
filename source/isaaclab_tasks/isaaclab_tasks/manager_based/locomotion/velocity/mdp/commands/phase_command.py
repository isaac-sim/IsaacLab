# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING
import torch

from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg, CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    

class PhaseCommand(CommandTerm):
    """
    A phase command that repeats from [0, 1].
    Locomotion gait period is randomized between a given range called gait_period. 
    """

    def __init__(self, cfg: PhaseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.step_dt = env.step_dt
        self.phase_dt = torch.empty(self.num_envs, device=self.device).uniform_(
            cfg.gait_period[0],
            cfg.gait_period[1]
        )
        self.phase_increment_per_step = self.step_dt / self.phase_dt

    def _update_command(self) -> None:
        self.phase += self.phase_increment_per_step  # advance by an amount of a step
        self.phase = torch.fmod(self.phase, 1.0)  # regulate to [0, 1]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # resample phase freq and dt
        self.phase[env_ids] = 0.0  # reset phase to 0
        self.phase_dt[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.gait_period[0],
            self.cfg.gait_period[1]
        )
        self.phase_increment_per_step[env_ids] = self.step_dt / self.phase_dt[env_ids]

    def _update_metrics(self):
        pass  # no metrics

    @property
    def command(self) -> torch.Tensor:
        return self.phase

@configclass
class PhaseCommandCfg(CommandTermCfg):
    class_type: type = PhaseCommand
    gait_period: tuple[float, float] = (0.4, 1.0)