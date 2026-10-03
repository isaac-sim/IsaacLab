# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rizon–Sharpa teaspoon manipulation with Newton granular coffee."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Rizon-Sharpa-Coffee",
    entry_point=f"{__name__}.coffee_env:CoffeeEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.coffee_env_cfg:CoffeeEnvCfg"},
)
