# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kuka-Allegro reinforcement-learning configuration for cube stacking."""

import gymnasium as gym

from . import agents


gym.register(
    id="IsaacContrib-Stack-Cube-KukaAllegro-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_rl_env_cfg:KukaAllegroCubeStackRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KukaAllegroStackPPORunnerCfg",
    },
    disable_env_checker=True,
)
