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
    id="Isaac-Velocity-Flat-Cassie-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CassieFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.CassieFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CassieFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.CassieFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Cassie-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.CassieRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.CassieRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Cassie-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.CassieRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.CassieRoughPPORunnerCfg,
    },
)
