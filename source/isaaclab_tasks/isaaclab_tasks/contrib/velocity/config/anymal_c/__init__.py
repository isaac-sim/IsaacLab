# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

_AGENTS_MODULE = f"{__name__}.agents"

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Velocity-Flat-AnymalC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalCFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerWithSymmetryCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Flat-AnymalC-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalCFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerWithSymmetryCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-AnymalC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AnymalCRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerWithSymmetryCfg"
        ),
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_rough_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-AnymalC-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AnymalCRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerWithSymmetryCfg"
        ),
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_rough_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_rough_ppo_cfg.yaml",
    },
)
