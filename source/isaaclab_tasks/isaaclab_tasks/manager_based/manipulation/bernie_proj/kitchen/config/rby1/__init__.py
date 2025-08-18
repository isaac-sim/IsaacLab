# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

"""
Task Environemnts
"""

# Suppose all your environments are defined in ycb_franka_env_cfg
# with class names like FrankaYCBAppleEnvCfg, etc.
# We list each environment in a simple structure:
YCB_ENVS = [
    ("Dishwasher", "RBY1KitchenDishwasherEnvCfg"),
    ("Door", "RBY1KitchenDoorEnvCfg"),
    # ("Freezer", "RBY1KitchenFreezerEnvCfg"),
    ("Microwave", "RBY1KitchenMicrowaveEnvCfg"),
]

for env_name, env_class in YCB_ENVS:
    gym.register(
        id=f"Kitchen-{env_name}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            # e.g. "env_cfg_entry_point": "my_module.ycb_franka_env_cfg:FrankaYCBAppleEnvCfg"
            "env_cfg_entry_point": f"{__name__}.kitchen_rby1_env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:YCBPPORunnerCfg",
        },
    )

gym.register(
    id="Kitchen-Freezer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.freezer_rby1_env_cfg:RBY1FreezerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KitchenPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)