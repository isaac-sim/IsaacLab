# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.orbit.utils import configclass

from .base_env_cfg import BaseEnvCfg
from .ui import RLTaskEnvWindow


@configclass
class RLTaskEnvCfg(BaseEnvCfg):
    """Configuration for a reinforcement learning environment."""

    # ui settings
    ui_window_class_type: type | None = RLTaskEnvWindow

    # general settings
    episode_length_s: float = MISSING
    """Duration of an episode (in seconds)."""

    # environment settings
    rewards: object = MISSING
    """Reward settings."""
    terminations: object = MISSING
    """Termination settings."""
    curriculum: object = MISSING
    """Curriculum settings."""
    commands: object = MISSING
    """Command settings."""
