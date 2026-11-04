# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load MicroDuck reference motions and support complete-clip playback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from ...data.motion_loader import MotionLoader
from ...mdp.commands import MotionCommand, MotionCommandCfg
from .motion_adapter import load_microduck_motion


class MicroDuckMotionCommand(MotionCommand):
    """Use source NPZ files directly with the native MicroDuck asset."""

    cfg: MicroDuckMotionCommandCfg

    def _load_motion(self) -> MotionLoader:
        return load_microduck_motion(
            self.cfg.motion_file,
            body_names=self.cfg.body_names,
            joint_names=self.robot.joint_names,
            device=self.device,
            body_com_pos_b=self.robot.data.body_com_pos_b.torch[0, self.body_indices],
            input_velocity_point=self.cfg.input_velocity_point,
        )

    def _adaptive_sampling(self, env_ids: Sequence[int] | slice) -> None:
        if self.cfg.start_at_first_frame:
            self.time_steps[env_ids] = 0
        else:
            super()._adaptive_sampling(env_ids)


@configclass
class MicroDuckMotionCommandCfg(MotionCommandCfg):
    """Source velocity convention and deterministic full-clip playback option."""

    class_type: type = MicroDuckMotionCommand
    input_velocity_point: Literal["link", "com"] = "link"
    """Original normalized MicroDuck NPZs store link-origin world velocities."""
    start_at_first_frame: bool = False
    """Disable adaptive start-frame sampling for complete-clip evaluation."""


def motion_complete(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """End a finite reference clip; configure this term as a timeout."""
    command: MicroDuckMotionCommand = env.command_manager.get_term(command_name)
    return command.time_steps >= command.motion.time_step_total - 1
