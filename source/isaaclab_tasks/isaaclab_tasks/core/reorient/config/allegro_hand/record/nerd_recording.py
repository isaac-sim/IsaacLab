# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent hand-state and joint-target exploration."""

import torch
from isaaclab_neural.record.recording import NeRDExplorationAction
from newton_neural.data import CommandBase, SamplerBase
from torch import Tensor


def sample_regimes(scales: tuple[float, ...], hand_count: int, low: Tensor, finger_joint_count: int) -> Tensor:
    """Sample ``[hand_count, joint_count]`` within plus or minus the scale of each finger's regime.

    Half of the hands draw one regime for all fingers, so the data covers both independent and shared regimes.
    """
    joint_count, device = len(low), low.device
    choice = torch.randint(len(scales), (hand_count, joint_count // finger_joint_count), device=device)
    together = torch.rand(hand_count, 1, device=device) < 0.5
    choice = torch.where(together, choice[:, :1], choice)
    scale = scales[choice].repeat_interleave(finger_joint_count, dim=1)
    return scale * (2.0 * torch.rand_like(scale) - 1.0)


class SamplerHandStates(SamplerBase):
    """Joint angles uniform within ``[low, high]`` [rad], and joint speeds in the regime of each finger."""

    def __init__(self, low: Tensor, high: Tensor, speeds: tuple[float, ...], finger_joint_count: int) -> None:
        self.low, self.high = low, high
        self.speeds = torch.tensor(speeds, device=low.device)
        self.finger_joint_count = finger_joint_count

    def sample(self, count: int) -> dict[str, Tensor]:
        angles = self.low + (self.high - self.low) * torch.rand(count, len(self.low), device=self.low.device)
        return {"joint_q": angles, "joint_qd": sample_regimes(self.speeds, count, self.low, self.finger_joint_count)}


class CommandHandTargets(CommandBase):
    """Joint targets offset from ``joint_q`` in the regime of each finger, kept within the joint limits.

    ``low``, ``high`` and ``world_count`` are set by the recorder when it starts the generator.
    """

    def __init__(self, reaches: tuple[float, ...], finger_joint_count: int) -> None:
        super().__init__()
        self.reaches, self.finger_joint_count = reaches, finger_joint_count

    def next(self, frames: dict[str, Tensor]) -> Tensor:
        offsets = sample_regimes(
            torch.as_tensor(self.reaches, device=self.low.device), self.world_count, self.low, self.finger_joint_count
        )
        return (frames["joint_q"] + offsets).clamp(self.low, self.high)


class AllegroHandNeRDExploration(NeRDExplorationAction):
    """Record the hand's free acceleration at independent states and targets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        low, high = self.binding.state_bounds()["joint_q"]
        self.states = SamplerHandStates(
            low - cfg.limit_margin, high + cfg.limit_margin, cfg.speeds, cfg.finger_joint_count
        )
        self.commands = CommandHandTargets(cfg.target_reaches, cfg.finger_joint_count)
        self.initialize_sampling()
