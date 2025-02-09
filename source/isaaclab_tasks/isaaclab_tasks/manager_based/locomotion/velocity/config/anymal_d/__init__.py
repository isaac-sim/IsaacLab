# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalDFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-D-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalDFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AnymalDRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-D-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AnymalDRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalDRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
