# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position Reach tasks on MDP2."""

import gymnasium as gym

for robot, module, agent_module in (("UR10", "ur10", "ur_10"), ("Franka", "franka", "franka")):
    gym.register(
        id=f"Isaac-Reach-{robot}-MDP2",
        entry_point=f"{__name__}.reach_env:ReachMDP2RLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{module}_env_cfg:{robot}ReachMDP2EnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"isaaclab_tasks.core.reach.config.{agent_module}.agents.rsl_rl_ppo_cfg:{robot}ReachPPORunnerCfg"
            ),
            "default_agent": "rsl_rl",
        },
    )
