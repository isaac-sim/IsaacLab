
import gymnasium as gym
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .camera_config import CameraBaseCfg, CameraPresets
from .robot_config import G1_29DOF_BODY_JOINT_INDICES, G1_DEX3_JOINT_INDICES, G1RobotPresets

__all__ = ["G1_29DOF_BODY_JOINT_INDICES", "G1_DEX3_JOINT_INDICES", "G1RobotPresets", "CameraBaseCfg", "CameraPresets"]


# Gymnasium registrations.
gym.register(
    id="IsaacContrib-Assemble-Trocar-G129-Dex3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.assemble_trocar.g129_dex3_env_cfg:G1AssembleTrocarEnvCfg",
    },
)

gym.register(
    id="IsaacContrib-Assemble-Trocar-G129-Dex3-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.assemble_trocar.g129_dex3_env_cfg:G1AssembleTrocarEvalEnvCfg",
    },
)
