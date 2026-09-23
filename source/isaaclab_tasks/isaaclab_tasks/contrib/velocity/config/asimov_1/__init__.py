# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asimov-1 velocity-tracking and adversarial motion-prior tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="IsaacContrib-AMP-Asimov1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Asimov1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1AMPRunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacContrib-AMP-Asimov1-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:Asimov1AmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1AMPRunnerCfg",
        "default_agent": "rsl_rl",
    },
)
