# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

_AGENTS_MODULE = f"{__name__}.agents"

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Velocity-Flat-UnitreeA1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{_AGENTS_MODULE}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Flat-UnitreeA1-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{_AGENTS_MODULE}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-UnitreeA1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_rough_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{_AGENTS_MODULE}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-UnitreeA1-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_MODULE}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_MODULE}:skrl_rough_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{_AGENTS_MODULE}:sb3_ppo_cfg.yaml",
    },
)
