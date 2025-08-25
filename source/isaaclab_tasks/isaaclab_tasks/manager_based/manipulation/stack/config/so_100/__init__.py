# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    stack_ik_rel_env_cfg,
    stack_joint_pos_env_cfg,
    stack_pink_ik_rel_env_cfg,
    stack_pink_ik_rel_visuomotor_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Stack-Cube-SO100-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_env_cfg.SO100CubeStackJointPosEnvCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##
gym.register(
    id="Isaac-Stack-Cube-SO100-Pink-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_pink_ik_rel_env_cfg.SO100CubeStackPinkIKRelEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-SO100-Pink-IK-Rel-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_pink_ik_rel_visuomotor_env_cfg.SO100CubeStackPinkIKRelVisuomotorEnvCfg,
    },
    disable_env_checker=True,
)
