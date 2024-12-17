# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid Standup environment.
"""

import gymnasium as gym
from .standup_env import StandUpEnv, SigmabanEnvCfg

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-SigmabanStandUp-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.humanoid_standup:StandUpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SigmabanEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
