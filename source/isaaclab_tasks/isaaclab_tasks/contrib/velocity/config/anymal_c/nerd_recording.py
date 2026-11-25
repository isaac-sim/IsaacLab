# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exploration of ANYmal C under held joint targets."""

import torch
from isaaclab_neural.record.recording import NeRDExplorationAction
from newton_neural.data import CommandHeld, SamplerFloatingBase


class AnymalCNeRDExploration(NeRDExplorationAction):
    """Drop moving robots from tilted poses and keep falls in the dataset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        low, high = self.binding.state_bounds()["joint_q"]
        place = (low[:3] + high[:3]) / 2
        upright = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.states = SamplerFloatingBase(
            torch.cat((place, upright, self.default_joint_pos)),
            position=((0.0, 0.0, 0.0), (0.0, 0.0, cfg.drop)),
            tilt=cfg.tilt,
            linear_speed=cfg.base_speed[0],
            angular_speed=cfg.base_speed[1],
            joint_reach=cfg.joint_reach,
            joint_speed=cfg.joint_speed,
        )
        # The action term's declared clipping bounds are the sampling bounds too.
        bounds = torch.empty_like(self._clip[0])
        bounds[self.joint_order] = self._clip[0]
        self.commands = CommandHeld(cfg.hold_frames, low=bounds[:, 0], high=bounds[:, 1])
        self.initialize_sampling()
