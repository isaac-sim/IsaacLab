# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright 2026 the SuperMimic authors.

"""Motion inputs and reset settings for model-based tracking."""

from dataclasses import MISSING, dataclass

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets.motion_loader import CsvMotionCfg


@dataclass
class ReturnSamplerCfg:
    gamma: float = 0.99
    """Discount of the measured return; sets v_max = max_step_reward / (1 - gamma)."""
    horizon: int = 20
    """Steps per update. Each env's last `horizon` steps turn into returns that
  bootstrap from V where the window ends."""
    memory: int = 256
    """Most samples a bin's V remembers: V is the exact mean of a bin's first
  `memory` samples; after that, earlier samples count as `memory` samples against
  each update's new ones, so V follows the current policy."""
    steepness: float = 25.0
    """Softmax gain on V / v_max."""
    explore: float = 0.5
    """Share of resets spread uniformly over all bins."""
    keep: float = 0.1
    """Share of resets on bins in proportion to the summed square of the focus each
  received: concentrated focus earns keep, focus spread thin earns almost none."""


@configclass
class ReferenceCommandCfg(CommandTermCfg):
    class_type = "isaaclab_tasks.core.motion_tracking.mdp.model_based.commands:MotionCommand"
    resampling_time_range = (1.0e9, 1.0e9)
    motion_dir: str = MISSING
    motion_shard_index: int = 0
    motion_shard_count: int = 1
    joint_names: list[str] = MISSING
    tracked_bodies: list[str] = MISSING
    source_fps: float = 30.0
    csv: CsvMotionCfg = CsvMotionCfg()
    context_body: str = "torso_link"
    reset_noise: bool = True
    evaluation: bool = False
    sampling_mode: str = "external"
    representation: str = "gencoord"
    actor_obs: str = "proprio"
    return_sampler: ReturnSamplerCfg = ReturnSamplerCfg()
    friction_bodies: list[str] = [".*ankle_roll_link"]
