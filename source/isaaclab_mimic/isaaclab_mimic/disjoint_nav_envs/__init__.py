# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Disjoint Navigation."""

import gymnasium as gym

from .g1_disjoint_nav_env import G1DisjointNavEnvCfg

gym.register(
    id="Isaac-G1-Disjoint-Navigation",
    entry_point="isaaclab_mimic.disjoint_nav_envs.g1_disjoint_nav_env:G1DisjointNavEnv",
    kwargs={
        "env_cfg_entry_point": G1DisjointNavEnvCfg,
    },
    disable_env_checker=True,
)
