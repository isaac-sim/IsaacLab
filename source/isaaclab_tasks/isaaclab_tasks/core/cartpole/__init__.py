# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.

This package consolidates the direct-workflow and manager-based-workflow
cartpole tasks. Module files carry a ``_direct_`` or ``_manager_`` infix to
disambiguate the two workflows within the flat package layout.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments -- direct workflow.
##

gym.register(
    id="Isaac-Cartpole-Direct",
    entry_point=f"{__name__}.cartpole_direct_env:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_direct_env_cfg:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_direct_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_direct_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_direct_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_direct_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Camera-Direct",
    entry_point=f"{__name__}.cartpole_direct_camera_presets_env:CartpoleCameraPresetsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_direct_camera_presets_env_cfg:CartpoleCameraPresetsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_direct_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_direct_camera_ppo_cfg.yaml",
    },
)

##
# Register Gym environments -- manager-based workflow.
##

gym.register(
    id="Isaac-Cartpole",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_manager_env_cfg:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_manager_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_manager_ppo_cfg:CartpolePPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_manager_ppo_cfg:CartpolePPORunnerWithSymmetryCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_manager_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_manager_ppo_cfg.yaml",
    },
)

# Canonical perception task -- selects observation pipeline (raw RGB, raw depth,
# ResNet18 features, Theia-Tiny features) via the preset CLI (#5587). Two
# rl_games agent entry points cover the image-policy and feature-policy yamls;
# pick via ``--agent rl_games_cfg_entry_point`` (image, default) or
# ``--agent rl_games_feature_cfg_entry_point`` (pretrained-feature).
gym.register(
    id="Isaac-Cartpole-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_manager_camera_env_cfg:CartpoleCameraPresetsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_manager_camera_ppo_cfg.yaml",
        "rl_games_feature_cfg_entry_point": f"{agents.__name__}:rl_games_manager_feature_ppo_cfg.yaml",
    },
)
