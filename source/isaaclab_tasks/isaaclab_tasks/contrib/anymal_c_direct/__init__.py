# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

_AGENTS_MODULE = f"{__name__}.agents"

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Velocity-Flat-AnymalC-Direct",
    entry_point=f"{__name__}.anymal_c_env:AnymalCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_c_env_cfg:AnymalCFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-AnymalC-Direct",
    entry_point=f"{__name__}.anymal_c_env:AnymalCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_c_env_cfg:AnymalCRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_rough_ppo_cfg.yaml",
    },
)
