# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck motion tracking with foot and optional head position rewards."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Motion-Tracking-MicroDuck-Feet",
    entry_point="isaaclab_tasks.core.motion_tracking.motion_tracking_env:MotionTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.feet_env_cfg:MicroDuckFeetTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MicroDuckMotionTrackingPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Motion-Tracking-MicroDuck-Feet-Head",
    entry_point="isaaclab_tasks.core.motion_tracking.motion_tracking_env:MotionTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.feet_head_env_cfg:MicroDuckFeetHeadTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MicroDuckMotionTrackingPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
