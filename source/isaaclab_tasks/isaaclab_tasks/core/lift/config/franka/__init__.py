# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

_AGENTS_MODULE = f"{__name__}.agents"

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Lift-Cube-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{_AGENTS_MODULE}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lift-Cube-Franka-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{_AGENTS_MODULE}:sb3_ppo_cfg.yaml",
    },
)
