# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Catheter RL environments."""

import gymnasium as gym

from .catheter_state_env import CatheterStateEnv, CatheterStateEnvCfg

gym.register(
    id="CatheterState-v0",
    entry_point="isaaclab_newton.envs.catheter_state_env:CatheterStateEnv",
    disable_env_checker=True,
    kwargs={"cfg": CatheterStateEnvCfg()},
)

__all__ = [
    "CatheterStateEnv",
    "CatheterStateEnvCfg",
]
