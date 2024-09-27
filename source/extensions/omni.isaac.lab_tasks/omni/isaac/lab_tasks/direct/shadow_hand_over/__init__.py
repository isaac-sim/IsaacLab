# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ShadowHand Over environment.
"""

import gymnasium as gym

from . import agents
from .shadow_hand_over_env import ShadowHandOverEnv
from .shadow_hand_over_env_cfg import ShadowHandOverEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Shadow-Hand-Over-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.shadow_hand_over:ShadowHandOverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandOverEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
