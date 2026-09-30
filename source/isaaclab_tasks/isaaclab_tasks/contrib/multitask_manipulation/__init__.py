# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contributed heterogeneous fixed-arm manipulation environments."""

import gymnasium as gym

from . import agents


gym.register(
    id="IsaacContrib-Multitask-Manipulation",
    entry_point=f"{__name__}.multitask_env:MultitaskManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multitask_env_cfg:MultitaskManipulationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultitaskManipulationPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
