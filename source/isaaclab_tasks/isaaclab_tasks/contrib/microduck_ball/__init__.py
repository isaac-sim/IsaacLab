# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck balancing and rolling on a ball, in four stages: fixed ball, free ball, velocity command, robustness."""

import gymnasium as gym

_STAGES = ("Fixed", "Balance", "Roll", "Robust")

for _stage in _STAGES:
    gym.register(
        id=f"IsaacContrib-Velocity-Ball-{_stage}-MicroDuck",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ball_env_cfg:MicroDuckBall{_stage}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckBall{_stage}PPORunnerCfg",
        },
    )
