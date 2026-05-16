# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from isaaclab_tasks.utils import deprecated_task_alias

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

# Canonical perception task -- selects observation pipeline (raw RGB, raw depth,
# ResNet18 features, Theia-Tiny features) via the preset CLI (#5587). Two
# rl_games agent entry points cover the image-policy and feature-policy yamls;
# pick via ``--agent rl_games_cfg_entry_point`` (image, default) or
# ``--agent rl_games_feature_cfg_entry_point`` (pretrained-feature). Old
# per-pipeline IDs below remain registered for one release as deprecation
# shims pointing at this task.
_CAMERA_CFG_PATH = f"{__name__}.cartpole_camera_env_cfg:CartpoleCameraPresetsEnvCfg"

gym.register(
    id="Isaac-Cartpole-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _CAMERA_CFG_PATH,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "rl_games_feature_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)

# -- Deprecated aliases --------------------------------------------------------

gym.register(
    id="Isaac-Cartpole-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-RGB-v0",
            "--task=Isaac-Cartpole-Camera-v0 presets=rgb",
            _CAMERA_CFG_PATH,
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-Depth-v0",
            "--task=Isaac-Cartpole-Camera-v0 presets=depth",
            _CAMERA_CFG_PATH,
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-RGB-ResNet18-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-RGB-ResNet18-v0",
            "--task=Isaac-Cartpole-Camera-v0 presets=resnet18 --agent=rl_games_feature_cfg_entry_point",
            _CAMERA_CFG_PATH,
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-RGB-TheiaTiny-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            "Isaac-Cartpole-RGB-TheiaTiny-v0",
            "--task=Isaac-Cartpole-Camera-v0 presets=theia_tiny --agent=rl_games_feature_cfg_entry_point",
            _CAMERA_CFG_PATH,
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)
