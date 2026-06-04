# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (experimental manager-based entry point).
"""

import gymnasium as gym

# Reuse agent configs from the stable task package.
from isaaclab_tasks.core.manager_ant import agents

# Warp tasks reuse the stable env cfgs; only physics (presets=newton_mjwarp)
# and MDP twins are swapped at construction (see adapt_cfg_for_warp).
_stable_pkg = agents.__name__.rsplit(".", 1)[0]

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ant-Warp-v0",
    entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_stable_pkg}.ant_env_cfg:AntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
