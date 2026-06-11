# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Allegro hand in-hand cube reorientation tasks (direct and manager-based workflows)."""

import gymnasium as gym

from isaaclab_tasks.core.reorient.config.allegro_hand import agents

##
# Register Gym environments -- manager-based workflow.
##

gym.register(
    id="Isaac-Reorient-Cube-Allegro",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_hand_manager_env_cfg:AllegroCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_manager_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_manager_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reorient-Cube-Allegro-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_hand_manager_env_cfg:AllegroCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_manager_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_manager_ppo_cfg.yaml",
    },
)

##
# Register Gym environments -- direct workflow.
##

gym.register(
    id="Isaac-Reorient-Cube-Allegro-Direct",
    entry_point="isaaclab_tasks.core.reorient.reorient_direct_env:ReorientDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_hand_direct_env_cfg:AllegroHandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_direct_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_direct_ppo_cfg.yaml",
    },
)
