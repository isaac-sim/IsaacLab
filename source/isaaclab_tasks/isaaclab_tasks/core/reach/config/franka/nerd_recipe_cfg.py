# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Settings of the neural robot dynamics model behind ``physics=newton_nerd`` of the Franka reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_neural import NeRDRecipeCfg, NeRDRecordCfg, NeRDTrainCfg

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .nerd_recipe import FrankaReachNeRDRecipe


@configclass
class FrankaReachNeRDRecipeCfg(NeRDRecipeCfg):
    """FlashNeRD of the arm with its joint drives, in free space.

    One frame is one control period. Short episodes follow the reset cadence of the task, so the model also sees
    the first frames after a reset.
    """

    class_type: type[FrankaReachNeRDRecipe] | str = "{DIR}.nerd_recipe:FrankaReachNeRDRecipe"

    record: NeRDRecordCfg = NeRDRecordCfg(num_envs=4096, num_frames=2048, episode_frames=(17, 64))
    train: NeRDTrainCfg = NeRDTrainCfg(
        max_steps=24_000, window=16, batch_size=1024, learning_rate=2.0e-3, noise=0.1, horizons=(1, 30)
    )

    arm_joint_count: int = 7
    """Joints of the arm. They come first, the two gripper fingers after them."""

    state_speed: float = 1.0
    """Largest joint speed an episode starts with [rad/s]."""

    target_margin: float = 0.5
    """How far a joint target may lie past a joint limit [rad]."""

    hold_frames: tuple[int, int] = (1, 60)
    """Shortest and longest hold of a constant target [frames]."""

    explore_reach: float = 1.0
    """Offset of a target from the joint position, as the actions of an untrained policy scatter [rad]."""

    refine_reach: float = 0.3
    """The same for the smaller actions of a trained policy [rad]."""

    drift_timescale: float = 100.0
    """Time scale of slowly drifting targets [s]."""
