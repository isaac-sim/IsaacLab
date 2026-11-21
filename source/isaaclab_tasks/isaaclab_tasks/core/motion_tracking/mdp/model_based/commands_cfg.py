# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion inputs and reset settings for model-based tracking."""

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets.motion_loader import CsvMotionCfg


@configclass
class ReferenceCommandCfg(CommandTermCfg):
    class_type = "isaaclab_tasks.core.motion_tracking.mdp.model_based.commands:MotionCommand"
    resampling_time_range = (1.0e9, 1.0e9)
    motion_dir: str = MISSING
    motion_shard_index: int = 0
    motion_shard_count: int = 1
    kinematics_path: str = MISSING
    joint_names: list[str] = MISSING
    tracked_bodies: list[str] = MISSING
    source_fps: float = 30.0
    csv: CsvMotionCfg = CsvMotionCfg()
    context_body: str = "torso_link"
    reset_noise: bool = True
    evaluation: bool = False
    sampling_mode: str = "external"
    friction_bodies: list[str] = [".*ankle_roll_link"]
