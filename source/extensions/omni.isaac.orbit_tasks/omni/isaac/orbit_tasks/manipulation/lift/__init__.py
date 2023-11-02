# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment for lifting an object with fixed-base robot.
"""

import gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Lift-Franka-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_env_cfg:LiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LIFT_RSL_RL_PPO_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}.robomimic:bc.json",
        "robomimic_bcq_cfg_entry_point": f"{agents.__name__}.robomimic:bcq.json",
    },
)
