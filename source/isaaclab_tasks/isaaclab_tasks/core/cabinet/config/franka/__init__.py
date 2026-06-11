# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from isaaclab_tasks.core.cabinet.config.franka import agents

##
# Register Gym environments -- manager-based workflow.
##

gym.register(
    id="Isaac-Open-Drawer-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_manager_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_manager_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Open-Drawer-Franka-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCabinetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_manager_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_manager_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Register Gym environments -- direct workflow.
##

gym.register(
    id="Isaac-Open-Drawer-Franka-Direct",
    entry_point="isaaclab_tasks.core.cabinet.cabinet_direct_env:CabinetDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cabinet_direct_env_cfg:FrankaCabinetDirectEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_direct_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_direct_ppo_cfg.yaml",
    },
)
