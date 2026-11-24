# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Settings of the neural robot dynamics model behind ``physics=newton_mjwarp_nerd`` of the Allegro reorient task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_neural import NeRDRecipeCfg, NeRDRecordCfg, NeRDTrainCfg

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .nerd_recipe import AllegroHandNeRDRecipe


@configclass
class AllegroHandNeRDRecipeCfg(NeRDRecipeCfg):
    """FlashNeRD of the free acceleration of the hand, which MuJoCo Warp completes with the contact of the cube.

    One frame is one solver step of MuJoCo Warp. Every sample is independent, a random state and a random target.
    Every finger is in one of three regimes, fast exploration, manipulation of the cube, or holding near rest.
    """

    class_type: type[AllegroHandNeRDRecipe] | str = "{DIR}.nerd_recipe:AllegroHandNeRDRecipe"

    physics_preset: str = "newton_mjwarp_nerd"
    record: NeRDRecordCfg = NeRDRecordCfg(num_envs=16384, num_frames=256, episode_frames=(1, 1))
    # Free acceleration labels belong to the recorded states, so the inputs take no noise, and one frame is a sample.
    train: NeRDTrainCfg = NeRDTrainCfg(
        max_steps=60_000, window=0, batch_size=8192, learning_rate=1.0e-3, noise=0.0, horizons=()
    )

    limit_margin: float = 0.15
    """How far a joint may start past a joint limit [rad]."""

    speeds: tuple[float, ...] = (30.0, 3.0, 0.3)
    """Largest joint speed in each regime [rad/s]."""

    target_reaches: tuple[float, ...] = (2.0, 0.3, 0.03)
    """Largest offset of a target from the joint position in each regime [rad]."""

    finger_joint_count: int = 4
    """Joints of one finger, which are contiguous in the joint order."""

    constraint_count: int = 256
    """Constraint rows of MuJoCo Warp per hand, enough for random poses that touch themselves."""

    backbone_spans: tuple[int, ...] = (1, 1, 1, 1)
    """Attention spans of the backbone. One frame of history, since the free acceleration follows from state and
    control."""
