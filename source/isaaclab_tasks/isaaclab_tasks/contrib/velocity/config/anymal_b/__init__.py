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
    id="IsaacContrib-Velocity-Flat-AnymalB",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalBFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalBFlatPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalBFlatPPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Flat-AnymalB-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalBFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalBFlatPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalBFlatPPORunnerWithSymmetryCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-AnymalB",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AnymalBRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalBRoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalBRoughPPORunnerWithSymmetryCfg"
        ),
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-AnymalB-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AnymalBRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalBRoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalBRoughPPORunnerWithSymmetryCfg"
        ),
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_rough_ppo_cfg.yaml",
    },
)
