# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Anymal-B-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.AnymalBFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalBFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-B-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.AnymalBFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalBFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-B-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.AnymalBRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalBRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-B-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.AnymalBRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalBRoughPPORunnerCfg,
    },
)
