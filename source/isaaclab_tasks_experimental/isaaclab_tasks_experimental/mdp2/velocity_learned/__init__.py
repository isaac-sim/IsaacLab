# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP2 flat velocity tracking tasks with learned actuators."""

import gymnasium as gym

for prefix, robot, source in (
    ("Isaac", "AnymalD", "core.velocity.config.anymal_d"),
    ("IsaacContrib", "AnymalB", "contrib.velocity.config.anymal_b"),
    ("IsaacContrib", "AnymalC", "contrib.velocity.config.anymal_c"),
    ("IsaacContrib", "UnitreeGo1", "contrib.velocity.config.go1"),
):
    gym.register(
        id=f"{prefix}-Velocity-Flat-{robot}-MDP2",
        entry_point="isaaclab_experimental.envs:MDP2RLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_cfg:{robot}FlatMDP2EnvCfg",
            "rsl_rl_cfg_entry_point": f"isaaclab_tasks.{source}.agents.rsl_rl_ppo_cfg:{robot}FlatPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
