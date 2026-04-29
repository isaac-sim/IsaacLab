# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Deploy-CableInsertion-Rizon4s-Grav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sGravCableInsertionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravCableInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-CableInsertion-Rizon4s-Grav-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sGravCableInsertionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravCableInsertionRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-CableInsertion-Rizon4s-Grav-ROS-Inference-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:Rizon4sGravCableInsertionROSInferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravCableInsertionRNNPPORunnerCfg",
    },
)
