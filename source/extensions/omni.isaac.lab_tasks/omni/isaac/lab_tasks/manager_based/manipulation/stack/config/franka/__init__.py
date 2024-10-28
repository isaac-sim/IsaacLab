# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

##
# Register Gym environments.
##

task_entry = "omni.isaac.lab_tasks.manager_based.manipulation.stack.config.franka"

##
# Joint Position Control
##

gym.register(
    id="Isaac-Stack-Cube-Franka-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.stack_joint_pos_env_cfg:FrankaCubeStackEnvCfg",
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.stack_ik_rel_env_cfg:FrankaCubeStackEnvCfg",
    },
    disable_env_checker=True,
)
