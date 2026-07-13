# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).

This package consolidates the direct-workflow and manager-based-workflow ant
tasks. Module files carry a ``_direct_`` or ``_manager_`` infix to disambiguate
the two workflows within the flat package layout.
"""

import gymnasium as gym

_AGENTS_MODULE = f"{__name__}.agents"

##
# Register Gym environments -- direct workflow.
##

gym.register(
    id="Isaac-Ant-Direct",
    entry_point=f"{__name__}.ant_direct_env:AntEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_direct_env_cfg:AntEnvCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_direct_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AntDirectPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_direct_ppo_cfg.yaml",
    },
)

##
# Register Gym environments -- manager-based workflow.
##

gym.register(
    id="Isaac-Ant",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_manager_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_MODULE}:rl_games_manager_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_manager_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{_AGENTS_MODULE}:sb3_manager_ppo_cfg.yaml",
    },
)
