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
from .cartpole_camera_env_cfg import (
    CartpoleDepthCameraEnvCfg,
    CartpoleResNet18CameraEnvCfg,
    CartpoleRGBCameraEnvCfg,
    CartpoleTheiaTinyCameraEnvCfg,
)

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
gym.register(
    id="Isaac-Cartpole-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_camera_env_cfg:CartpoleCameraPresetsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "rl_games_feature_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)

# -- Deprecated aliases --------------------------------------------------------
# Each retired task ID returns its historical per-variant cfg via ``cfg_factory``
# so the retired ID stays bit-for-bit identical to develop. The OLD subclasses
# (``Cartpole{RGB,Depth,RGB-ResNet18,RGB-TheiaTiny}CameraEnvCfg``) in
# ``cartpole_camera_env_cfg.py`` are kept for one release alongside the
# consolidated ``CartpoleCameraPresetsEnvCfg`` and will be removed together
# with these task IDs.

gym.register(
    id="Isaac-Cartpole-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": deprecated_task_alias(
            old_task_id="Isaac-Cartpole-RGB-v0",
            new_command=["--task=Isaac-Cartpole-Camera-v0", "presets=rgb"],
            cfg_factory=lambda: CartpoleRGBCameraEnvCfg(),
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
            old_task_id="Isaac-Cartpole-Depth-v0",
            new_command=["--task=Isaac-Cartpole-Camera-v0", "presets=depth"],
            cfg_factory=lambda: CartpoleDepthCameraEnvCfg(),
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
            old_task_id="Isaac-Cartpole-RGB-ResNet18-v0",
            new_command=[
                "--task=Isaac-Cartpole-Camera-v0",
                "--agent=rl_games_feature_cfg_entry_point",
                "presets=resnet18",
            ],
            cfg_factory=lambda: CartpoleResNet18CameraEnvCfg(),
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
            old_task_id="Isaac-Cartpole-RGB-TheiaTiny-v0",
            new_command=[
                "--task=Isaac-Cartpole-Camera-v0",
                "--agent=rl_games_feature_cfg_entry_point",
                "presets=theia_tiny",
            ],
            cfg_factory=lambda: CartpoleTheiaTinyCameraEnvCfg(),
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_feature_ppo_cfg.yaml",
    },
)
