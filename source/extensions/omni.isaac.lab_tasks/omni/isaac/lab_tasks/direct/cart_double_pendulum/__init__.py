# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Inverted Double Pendulum on a Cart balancing environment.
"""

import gymnasium as gym

from . import agents
from .cart_double_pendulum_env import CartDoublePendulumEnv, CartDoublePendulumEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Cart-Double-Pendulum-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.cart_double_pendulum:CartDoublePendulumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartDoublePendulumEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
