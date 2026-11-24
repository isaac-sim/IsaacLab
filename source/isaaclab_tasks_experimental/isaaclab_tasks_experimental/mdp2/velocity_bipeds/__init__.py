# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP2 flat velocity tracking with Unitree G1 and H1."""

import gymnasium as gym

for robot, runner_module in (("G1", "g1"), ("H1", "h1")):
    gym.register(
        id=f"Isaac-Velocity-Flat-{robot}-MDP2",
        entry_point="isaaclab_experimental.envs:MDP2RLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_cfg:{robot}FlatMDP2EnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"isaaclab_tasks.core.velocity.config.{runner_module}.agents.rsl_rl_ppo_cfg:{robot}FlatPPORunnerCfg"
            ),
            "default_agent": "rsl_rl",
        },
    )
