# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka suction sorting using the experimental upstream Newton seal model."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Conveyor-Suction-Sorting",
    entry_point=f"{__name__}.env:SuctionSortingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SuctionSortingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:SuctionSortingPPORunnerCfg",
    },
)
