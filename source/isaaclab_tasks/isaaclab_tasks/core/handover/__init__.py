# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ShadowHand Over environment.
"""

import gymnasium as gym

_AGENTS_MODULE = f"{__name__}.agents"

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Shadow-Handover-Direct",
    entry_point=f"{__name__}.handover_env:HandoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handover_env_cfg:HandoverEnvCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_mappo_cfg.yaml",
    },
)
