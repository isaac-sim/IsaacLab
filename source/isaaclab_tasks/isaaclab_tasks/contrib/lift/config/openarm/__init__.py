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
    id="IsaacContrib-Lift-Cube-OpenArm",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:OpenArmLiftCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-Lift-Cube-OpenArm-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmCubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:OpenArmLiftCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
