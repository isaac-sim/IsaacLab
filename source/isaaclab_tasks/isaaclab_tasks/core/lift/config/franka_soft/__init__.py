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
    id="Isaac-Lift-Soft-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_soft_env_cfg:FrankaSoftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:FrankaDeformablePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Lift-Cloth-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_cloth_env_cfg:FrankaClothEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:FrankaDeformablePPORunnerCfg",
    },
)
