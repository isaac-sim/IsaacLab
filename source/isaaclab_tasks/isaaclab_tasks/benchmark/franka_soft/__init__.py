# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Lift-Cloth-Franka-Camera-Benchmark",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_cloth_camera_env_cfg:FrankaClothCameraEnvCfg",
    },
)

gym.register(
    id="Isaac-Lift-Soft-Franka-Camera-Benchmark",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_soft_camera_env_cfg:FrankaSoftCameraEnvCfg",
    },
)
