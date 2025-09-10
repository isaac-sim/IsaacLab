# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

##
# Register Gym environments.
##

##
# RMPFlow (with Joint Limit Constraint and Obstacle Avoidance) for Galbot Single Arm Cube Stack Task
# you can use for both absolute and relative mode, by given the USE_RELATIVE_MODE environment variable
##
gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_rmp_rel_env_cfg:RmpFlowGalbotLeftArmCubeStackEnvCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_rmp_rel_env_cfg:RmpFlowGalbotRightArmCubeStackEnvCfg",
    },
    disable_env_checker=True,
)


##
# Visuomotor Task for Galbot Left ArmCube Stack Task
##
gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_rmp_rel_env_cfg:RmpFlowGalbotLeftArmCubeStackVisuomotorEnvCfg",
    },
    disable_env_checker=True,
)

##
# Policy Close-loop Evaluation Task for Galbot Left Arm Cube Stack Task (in Joint Space)
##
gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-Joint-Position-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.stack_rmp_rel_env_cfg:GalbotLeftArmJointPositionCubeStackVisuomotorEnvCfg_PLAY"
        ),
    },
    disable_env_checker=True,
)

##
# Policy Close-loop Evaluation Task for Galbot Left Arm Cube Stack Task (in Task Space)
##
gym.register(
    id="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-RmpFlow-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_rmp_rel_env_cfg:GalbotLeftArmRmpFlowCubeStackVisuomotorEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
