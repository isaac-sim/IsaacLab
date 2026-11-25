# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the tabletop scene with a Franka arm and Robotiq gripper."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-TableTop-PickPlace-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.heterogeneous_tasks.heterogeneous_tasks_env_cfg:TabletopPickPlaceEnvCfg"
        )
    },
    disable_env_checker=True,
)
