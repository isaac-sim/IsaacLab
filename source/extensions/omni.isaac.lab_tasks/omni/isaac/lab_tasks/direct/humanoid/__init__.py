# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents
from .humanoid_env import HumanoidEnv, HumanoidEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Humanoid-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.humanoid:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HumanoidEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
