# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Franka MPM pouring task."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Franka-Pour",
    entry_point="isaaclab_tasks.contrib.franka_pour.pour_env:FrankaPourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.franka_pour.pour_env_cfg:FrankaPourResetDatasetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaPourResetDatasetPPORunnerCfg",
    },
)
