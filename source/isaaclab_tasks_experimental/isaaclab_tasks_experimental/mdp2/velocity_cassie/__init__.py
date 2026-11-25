# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP2 flat-ground velocity tracking with Cassie."""

import gymnasium as gym

gym.register(
    id="Isaac-Velocity-Flat-Cassie-MDP2",
    entry_point="isaaclab_experimental.envs:MDP2RLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CassieFlatMDP2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.core.velocity.config.cassie.agents.rsl_rl_ppo_cfg:CassieFlatPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)
