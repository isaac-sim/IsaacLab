# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (experimental manager-based entry point).
"""

import gymnasium as gym

from isaaclab_experimental.envs.frontend import register_mdp_route

# The stable Ant task borrows Humanoid MDP terms, so its warp twins live in the
# experimental humanoid package.
register_mdp_route(
    "isaaclab_tasks.core.locomotion.ant",
    "isaaclab_tasks_experimental.manager_based.classic.humanoid.mdp",
)

# Reuse agent configs from the stable task package.
from isaaclab_tasks.core.locomotion.ant import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ant-Warp-v0",
    entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_manager_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_manager_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_manager_ppo_cfg.yaml",
    },
)
