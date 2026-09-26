# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka fruit preparation with a visual tap and physical lid assembly."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Franka-Smoothie",
    entry_point=f"{__name__}.smoothie_env:SmoothieBlenderEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.smoothie_env_cfg:FrankaSmoothieEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:FrankaSmoothiePPORunnerCfg",
    },
)
