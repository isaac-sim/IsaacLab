
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="Isaac-Flat-Franka-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_multitask_flat_env_cfg:FlatSingleRobotMultiTaskEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Flat-Multi-Robot-Multi-Task-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_multi_robot_task_flat_env_cfg:FlatMultiRobotMultiTaskEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Flat-Multi-Robot-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.demo_multi_robot_reach_env_cfg:MultiRobotReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MultiRobotReachPPORunnerCfg",
    },
    disable_env_checker=True,
)
