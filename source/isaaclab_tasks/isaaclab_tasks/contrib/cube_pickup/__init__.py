# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse cube pickup demonstrating a generated and validated reset-state buffer."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Cube-Pickup-Franka",
    entry_point=f"{__name__}.cube_pickup_env:CubePickupEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_pickup_env_cfg:CubePickupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:CubePickupPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
