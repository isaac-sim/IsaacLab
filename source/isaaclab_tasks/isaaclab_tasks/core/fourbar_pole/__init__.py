# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fourbar-pole swing-up environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments -- manager-based workflow.
##

gym.register(
    id="Isaac-Fourbar-Pole-Swingup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fourbar_pole_manager_env_cfg:FourbarPoleSwingupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_manager_ppo_cfg:FourbarPolePPORunnerCfg",
    },
)
