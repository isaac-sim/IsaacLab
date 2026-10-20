# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-Franka Newton shoelace task."""

import gymnasium as gym

from . import agents


gym.register(
    id="IsaacContrib-Shoelace-DualFranka",
    entry_point=f"{__name__}.shoelace_physics:create_shoelace_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shoelace_env_cfg:ShoelaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShoelacePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
