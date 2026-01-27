# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T3 Hexapod locomotion velocity tracking environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments for T3 Hexapod
##

# Flat terrain environments
gym.register(
    id="Isaac-Velocity-Flat-T3-Hexapod-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:T3HexapodFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T3HexapodFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-T3-Hexapod-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:T3HexapodFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T3HexapodFlatPPORunnerCfg",
    },
)

# Rough terrain environments
gym.register(
    id="Isaac-Velocity-Rough-T3-Hexapod-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:T3HexapodRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T3HexapodRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-T3-Hexapod-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:T3HexapodRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T3HexapodRoughPPORunnerCfg",
    },
)
