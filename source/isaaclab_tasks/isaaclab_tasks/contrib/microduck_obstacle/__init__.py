# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ToF obstacle avoidance using a frozen MicroDuck walking policy.

The walking tasks remain in :mod:`isaaclab_tasks.contrib.microduck`.
Select the backlash robot with ``presets=backlash``.
"""

import gymnasium as gym

gym.register(
    id="IsaacContrib-ObstacleAvoidance-Flat-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obstacle_avoidance_env_cfg:MicroDuckObstacleAvoidanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckObstacleAvoidancePPORunnerCfg",
    },
)
