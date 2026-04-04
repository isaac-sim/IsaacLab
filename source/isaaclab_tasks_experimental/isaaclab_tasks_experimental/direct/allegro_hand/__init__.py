# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Allegro Inhand Manipulation environment.
"""

import gymnasium as gym

##
# Register Gym environments.
##

inhand_task_entry = "isaaclab_tasks_experimental.direct.inhand_manipulation"
stable_agents = "isaaclab_tasks.direct.allegro_hand.agents"

gym.register(
    id="Isaac-Repose-Cube-Allegro-Direct-Warp-v0",
    entry_point=f"{inhand_task_entry}.inhand_manipulation_warp_env:InHandManipulationWarpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_hand_warp_env_cfg:AllegroHandWarpEnvCfg",
        "rl_games_cfg_entry_point": f"{stable_agents}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{stable_agents}.rsl_rl_ppo_cfg:AllegroHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{stable_agents}:skrl_ppo_cfg.yaml",
    },
)
