# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP2 registrations for the Ant and Humanoid manager tasks."""

import gymnasium as gym

for robot in ("Ant", "Humanoid"):
    gym.register(
        id=f"Isaac-{robot}-MDP2",
        entry_point="isaaclab_experimental.envs:MDP2RLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{robot.lower()}_env_cfg:{robot}MDP2EnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"isaaclab_tasks.core.locomotion.{robot.lower()}.agents.rsl_rl_ppo_cfg:{robot}PPORunnerCfg"
            ),
            "default_agent": "rsl_rl",
        },
    )
