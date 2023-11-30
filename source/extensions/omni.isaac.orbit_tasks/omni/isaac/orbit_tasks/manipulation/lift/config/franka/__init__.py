# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents, env_cfg

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Lift-Cube-Franka-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg.FrankaCubeLiftEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftCubePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-Franka-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg.FrankaCubeLiftEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftCubePPORunnerCfg,
    },
    disable_env_checker=True,
)
