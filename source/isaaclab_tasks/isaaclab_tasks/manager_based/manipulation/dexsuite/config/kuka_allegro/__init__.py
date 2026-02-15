# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
# gym.register(
#     id="Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroReorientEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
#     },
# )

# gym.register(
#     id="Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroReorientEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
#     },
# )

# Dexsuite Lift Environments
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)


# Vision-Based Environments
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dexsuite_kuka_allegro_vision_env_cfg:DexsuiteKukaAllegroLiftSingleCameraEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cnn_cfg:DexsuiteKukaAllegroPPOCNNRunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cnn_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dexsuite_kuka_allegro_vision_env_cfg:DexsuiteKukaAllegroLiftSingleCameraEnvCfg_PLAY"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cnn_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cnn_cfg:DexsuiteKukaAllegroPPOCNNRunnerCfg",
    },
)


# gym.register(
#     id="Isaac-Dexsuite-Kuka-Allegro-Lift-Duo-Camera-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": (
#             f"{__name__}.dexsuite_kuka_allegro_vision_env_cfg:DexsuiteKukaAllegroLiftDuoCameraEnvCfg"
#         ),
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_duo_camera_cfg:DexsuiteKukaAllegroPPORunnerDuoCameraCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cnn_cfg.yaml",
#     },
# )


# gym.register(
#     id="Isaac-Dexsuite-Kuka-Allegro-Lift-Duo-Camera-Play-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": (
#             f"{__name__}.dexsuite_kuka_allegro_vision_env_cfg:DexsuiteKukaAllegroLiftDuoCameraEnvCfg_PLAY"
#         ),
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cnn_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_duo_camera_cfg:DexsuiteKukaAllegroPPORunnerDuoCameraCfg"
#     },
# )
