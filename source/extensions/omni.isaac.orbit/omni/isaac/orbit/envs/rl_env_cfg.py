# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.orbit.utils import configclass

from .base_env_cfg import BaseEnvCfg


@configclass
class RLEnvCfg(BaseEnvCfg):
    """Configuration for a reinforcement learning environment."""

    # general settings
    episode_length_s: float = MISSING
    """Duration of an episode (in seconds)."""

    # environment settings
    rewards: object = MISSING
    """Reward settings."""
    terminations: object = MISSING
    """Termination settings."""
    randomization: object = MISSING
    """Randomization settings."""
    curriculum: object = MISSING
    """Curriculum settings."""
