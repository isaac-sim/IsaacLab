# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Franka berry-picking task."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Pick-Berry-Franka-IK-Rel-Newton",
    entry_point="isaaclab_tasks.contrib.franka_pick_berries.pick_berries_env:BerryPickEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaaclab_tasks.contrib.franka_pick_berries.pick_berries_env_cfg:BerryPickEnvCfg"},
)
