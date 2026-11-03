# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stand-up task for the MicroDuck biped."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-StandUp-Flat-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:standup_flat_env_cfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:standup_flat_agent_cfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="IsaacContrib-StandUp-Rough-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:standup_rough_env_cfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.contrib.microduck.preset_configs:standup_rough_agent_cfg",
        "default_agent": "rsl_rl",
    },
)
