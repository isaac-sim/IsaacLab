# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

##
# Register Gym environments.
##

stable_agents = "isaaclab_tasks.core.locomotion.ant.agents"

gym.register(
    id="Isaac-Ant-Direct-Warp-v0",
    entry_point=f"{__name__}.ant_env_warp:AntWarpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_warp_cfg:AntWarpEnvCfg",
        "rl_games_cfg_entry_point": f"{stable_agents}:rl_games_direct_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{stable_agents}.rsl_rl_ppo_cfg:AntDirectPPORunnerCfg",
        "skrl_cfg_entry_point": f"{stable_agents}:skrl_direct_ppo_cfg.yaml",
    },
)
