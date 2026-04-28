# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# Reuse agent configs from the stable task package.
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1 import agents

##
# Register Gym environments.
##

# gym.register(
#     id="Isaac-Velocity-Rough-H1-Warp-v0",
#     entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_env_cfg:H1RoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Rough-H1-Warp-Play-v0",
#     entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rough_env_cfg:H1RoughEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

gym.register(
    id="Isaac-Velocity-Flat-H1-Warp-v0",
    entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-H1-Warp-Play-v0",
    entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
