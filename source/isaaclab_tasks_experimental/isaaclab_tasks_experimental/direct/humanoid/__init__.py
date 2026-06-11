# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment.
"""

import gymnasium as gym

##
# Register Gym environments.
##

stable_agents = "isaaclab_tasks.core.locomotion.humanoid.agents"

gym.register(
    id="Isaac-Humanoid-Direct-Warp-v0",
    entry_point=f"{__name__}.humanoid_warp_env:HumanoidWarpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_warp_env_cfg:HumanoidWarpEnvCfg",
        "rl_games_cfg_entry_point": f"{stable_agents}:rl_games_direct_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{stable_agents}.rsl_rl_ppo_cfg:HumanoidDirectPPORunnerCfg",
        "skrl_cfg_entry_point": f"{stable_agents}:skrl_direct_ppo_cfg.yaml",
    },
)
