# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment.
"""

import gymnasium as gym
from .h1_env import H1Env, H1EnvCfg
from .sigmaban_env import SigmabanEnv, SigmabanEnvCfg

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Humanoid-Direct-v0",
    entry_point=f"{__name__}.humanoid_env:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_env:HumanoidEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-H1-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.humanoid:H1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Sigmaban-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.humanoid:SigmabanEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SigmabanEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
