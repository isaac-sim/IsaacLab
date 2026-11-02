# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 whole-body motion tracking."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Motion-Tracking-G1",
    entry_point="isaaclab_tasks.core.motion_tracking.motion_tracking_env:MotionTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1MotionTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionTrackingPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
