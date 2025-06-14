# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dextra Kuka Allegro environments.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Dextrah-Kuka-Allegro-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextrah_kuka_allegro_env_cfg:DextrahKukaAllegroEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DextrahKukaAllegroPPORunnerCfg"
    },
)
