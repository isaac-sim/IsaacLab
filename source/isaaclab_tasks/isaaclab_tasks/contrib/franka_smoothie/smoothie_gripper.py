# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Six-phase gripper targets preserving the existing filtered joint action."""

from dataclasses import field

import torch

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.franka_pour.mdp.actions import CurriculumGripperPositionAction
from isaaclab_tasks.contrib.franka_pour.mdp.actions_cfg import CurriculumGripperPositionActionCfg


class SmoothieGripperPositionAction(CurriculumGripperPositionAction):
    """Change physical finger targets [m] without clearing filter history."""

    def __init__(self, cfg, env):
        if cfg.open_positions_by_phase != {0: 0.015, 1: 0.04, 2: 0.04, 3: 0.04, 4: 0.04, 5: 0.04}:
            raise ValueError("Require declared six-phase opening targets.")
        if cfg.close_positions_by_phase != dict.fromkeys(range(6), 0.0):
            raise ValueError("Require six zero closure targets.")
        super().__init__(cfg, env)
        self._open_command = self._open_command.expand(self.num_envs, -1).clone()
        self._close_command = self._close_command.expand(self.num_envs, -1).clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply phase targets before the existing binary action filter."""
        phase = self._env.phase
        if (
            phase.shape != (self.num_envs,)
            or phase.dtype != torch.long
            or not bool(((phase >= 0) & (phase <= 5)).all())
        ):
            raise ValueError("Require one integer phase 0..5 per environment.")
        self._open_command.copy_(torch.where((phase == 0)[:, None], 0.015, 0.04).expand_as(self._open_command))
        self._close_command.zero_()
        super().process_actions(actions)


@configclass
class SmoothieGripperPositionActionCfg(CurriculumGripperPositionActionCfg):
    """Explicit per-finger phase targets [m]."""

    class_type: type[SmoothieGripperPositionAction] = SmoothieGripperPositionAction
    open_positions_by_phase: dict[int, float] = field(
        default_factory=lambda: {0: 0.015, 1: 0.04, 2: 0.04, 3: 0.04, 4: 0.04, 5: 0.04}
    )
    close_positions_by_phase: dict[int, float] = field(default_factory=lambda: dict.fromkeys(range(6), 0.0))
    neutral_position: float = 0.04
    close_position: float = 0.0
    default_position: float = 0.015
    alpha: float = 0.04
