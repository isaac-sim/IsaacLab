# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

# State Observation
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)

# Dexsuite Lift Environments
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Depth-TiledCamera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dexsuite_kuka_allegro_vision_env_cfg:DexsuiteKukaAllegroLiftDepthTiledCameraEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cnn_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Depth-TiledCamera-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dexsuite_kuka_allegro_vision_env_cfg:DexsuiteKukaAllegroLiftDepthTiledCameraEnvCfg_PLAY"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cnn_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Depth-RayCasterCamera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dexsuite_kuka_allegro_vision_env_cfg:DexsuiteKukaAllegroLiftDepthRayCasterCameraEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cnn_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Depth-RayCasterCamera-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dexsuite_kuka_allegro_vision_env_cfg:DexsuiteKukaAllegroLiftDepthRayCasterCameraEnvCfg_PLAY"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cnn_cfg.yaml",
    },
)

