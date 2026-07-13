# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

_AGENTS_MODULE = f"{__name__}.agents"

##
# Register Gym environments.
##


# Flexiv Rizon 4s
gym.register(
    id="IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sGearAssemblyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:Rizon4sGearAssemblyRNNPPORunnerCfg",
    },
)

# Flexiv Rizon 4s - Play / Debug (deterministic, no randomization)
gym.register(
    id="IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:Rizon4sGearAssemblyEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:Rizon4sGearAssemblyRNNPPORunnerCfg",
    },
)

# Flexiv Rizon 4s - ROS Inference
gym.register(
    id="IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav-ROS-Inference",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:Rizon4sGearAssemblyROSInferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:Rizon4sGearAssemblyRNNPPORunnerCfg",
    },
)
