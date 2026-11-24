# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registers the tasks with a Franka arm and a Robotiq 2F-85 gripper, and the environment whose environments each run
one of them."""

from importlib import import_module

import gymnasium as gym

CONFIG = "isaaclab_tasks.contrib.heterogeneous_tasks.heterogeneous_tasks_env_cfg"

gym.register(
    id="IsaacContrib-VLA-Heterogeneous-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{CONFIG}:VLAHeterogeneousEnvCfg"},
    disable_env_checker=True,
)
for index in range(256):
    gym.register(
        id=f"IsaacContrib-VLA-PickPlace{index:03d}-Franka",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": lambda index=index: import_module(CONFIG).VLAHeterogeneousEnvCfg(
            task_names=(f"PickPlace{index:03d}",)
        )},
        disable_env_checker=True,
    )
