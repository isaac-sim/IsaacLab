# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Humanoid locomotion environment (similar to OpenAI Gym Humanoid-v2).

This package consolidates the direct-workflow and manager-based-workflow humanoid tasks. Module
files carry a ``_direct_`` or ``_manager_`` infix to disambiguate the two workflows within the flat
package layout.
"""

import gymnasium as gym

from isaaclab_tasks.core.locomotion.humanoid import agents

##
# Register Gym environments -- direct workflow.
##

gym.register(
    id="Isaac-Humanoid-Direct",
    entry_point=f"{__name__}.humanoid_direct_env:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_direct_env_cfg:HumanoidEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_direct_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidDirectPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_direct_ppo_cfg.yaml",
    },
)

##
# Register Gym environments -- manager-based workflow.
##

gym.register(
    id="Isaac-Humanoid",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_manager_env_cfg:HumanoidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_manager_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_manager_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_manager_ppo_cfg.yaml",
    },
)
