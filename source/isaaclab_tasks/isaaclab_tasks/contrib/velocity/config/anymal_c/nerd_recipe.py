# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recipe of the neural robot dynamics model behind ``physics=newton_nerd`` of the ANYmal C flat task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab_neural import NeRDRecipe, RecordingSimulation
from newton_neural.contacts import ContactInputProbes
from newton_neural.data import CommandHeld, SamplerFloatingBase
from newton_neural.models import FlashNeRD

if TYPE_CHECKING:
    from .nerd_recipe_cfg import AnymalCFlatNeRDRecipeCfg


class AnymalCFlatNeRDRecipe(NeRDRecipe):
    """Drops the robot from tilted, moving poses and holds random joint targets around its default pose."""

    cfg: AnymalCFlatNeRDRecipeCfg

    def model(self, simulation: RecordingSimulation) -> FlashNeRD:
        feet = ContactInputProbes.from_model(
            simulation.model, simulation.system, bodies=self.cfg.foot_bodies, articulation=simulation.articulation
        )
        return FlashNeRD(simulation.system, contact_input=feet)

    def start_states(self, simulation: RecordingSimulation) -> SamplerFloatingBase:
        """Start above the ground, tilted and moving, around the default pose. The model does not depend on yaw."""
        cfg = self.cfg
        low, high = simulation.state_bounds()["joint_q"]
        place = (low[:3] + high[:3]) / 2  # Nominal position of the base
        upright = torch.tensor([0.0, 0.0, 0.0, 1.0], device=simulation.device)  # Quaternion, xyzw
        return SamplerFloatingBase(
            torch.cat((place, upright, simulation.default_joint_pos)),
            position=((0.0, 0.0, 0.0), (0.0, 0.0, cfg.drop)),
            tilt=cfg.tilt,
            linear_speed=cfg.base_speed[0],
            angular_speed=cfg.base_speed[1],
            joint_reach=cfg.joint_reach,
            joint_speed=cfg.joint_speed,
        )

    def joint_targets(self, simulation: RecordingSimulation) -> CommandHeld:
        default, reach = simulation.default_joint_pos, self.cfg.target_reach
        return CommandHeld(self.cfg.hold_frames, low=default - reach, high=default + reach)
