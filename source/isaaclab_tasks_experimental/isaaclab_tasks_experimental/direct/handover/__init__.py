# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Warp-first Direct variant of the Shadow handover task.

The environment reuses the stable configuration and agent definitions from
:mod:`isaaclab_tasks.core.handover`.
"""

import gymnasium as gym

##
# Register Gym environments.
##

stable_handover = "isaaclab_tasks.core.handover"
stable_handover_agents = f"{stable_handover}.agents"

gym.register(
    id="Isaac-Shadow-Handover-Direct-Warp-v0",
    entry_point="isaaclab_tasks_experimental.direct.handover.handover_warp_env:HandoverWarpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{stable_handover}.handover_env_cfg:HandoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{stable_handover_agents}.rsl_rl_ppo_cfg:HandoverPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{stable_handover_agents}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{stable_handover_agents}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{stable_handover_agents}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{stable_handover_agents}:skrl_mappo_cfg.yaml",
    },
)
