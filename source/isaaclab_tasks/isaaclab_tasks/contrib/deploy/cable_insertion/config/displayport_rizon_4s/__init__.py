# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

_INSERTION_ENV_ENTRY = (
    "isaaclab_tasks.contrib.deploy.cable_insertion.insertion_env:DisplayportInsertionEnv"
)

##
# Register Gym environments.
##

# Flexiv Rizon 4s - Joint space
gym.register(
    id="Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

# Flexiv Rizon 4s - Joint space Play
gym.register(
    id="Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-Play-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

# Flexiv Rizon 4s - Joint space without joint velocity
gym.register(
    id="Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionNoJointVelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

# Flexiv Rizon 4s - Joint space without joint velocity Play
gym.register(
    id="Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-Play-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionNoJointVelEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

# Flexiv Rizon 4s - Joint space ROS Inference without joint velocity
gym.register(
    id="Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.ros_inference_env_cfg:Rizon4sGravDisplayportInsertionNoJointVelROSInferenceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)

# Flexiv Rizon 4s - Joint space ROS Inference
gym.register(
    id="Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-ROS-Inference-v0",
    entry_point=_INSERTION_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:Rizon4sGravDisplayportInsertionROSInferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg",
    },
)
