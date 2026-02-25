# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# UR10e with 2F-140 gripper
gym.register(
    id="Isaac-Deploy-GearAssembly-UR10e-2F140-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F140GearAssemblyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F140GearAssemblyEnvCfg_PLAY",
    },
)

# UR10e with 2F-85 gripper
gym.register(
    id="Isaac-Deploy-GearAssembly-UR10e-2F85-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F85GearAssemblyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F85GearAssemblyEnvCfg_PLAY",
    },
)

# UR10e with 2F-140 gripper - ROS Inference
gym.register(
    id="Isaac-Deploy-GearAssembly-UR10e-2F140-ROS-Inference-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:UR10e2F140GearAssemblyROSInferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
    },
)

# UR10e with 2F-85 gripper - ROS Inference
gym.register(
    id="Isaac-Deploy-GearAssembly-UR10e-2F85-ROS-Inference-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:UR10e2F85GearAssemblyROSInferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
    },
)
