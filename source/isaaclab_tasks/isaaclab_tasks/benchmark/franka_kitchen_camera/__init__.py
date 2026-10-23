# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering-throughput benchmark: Franka Reach inside the Pixar Kitchen Set."""

import gymnasium as gym

gym.register(
    id="Isaac-Reach-Franka-Kitchen-Camera-Benchmark",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.franka_kitchen_camera_env_cfg:FrankaKitchenCameraEnvCfg"},
)
