# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering-throughput benchmark: the Rizon-Sharpa coffee demo's kitchen scene."""

import gymnasium as gym

gym.register(
    id="Isaac-RizonSharpa-Coffee-Kitchen-Camera-Benchmark",
    entry_point=f"{__name__}.coffee_kitchen_camera_env:CoffeeKitchenCameraEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.coffee_kitchen_camera_env_cfg:CoffeeKitchenCameraEnvCfg"},
)
