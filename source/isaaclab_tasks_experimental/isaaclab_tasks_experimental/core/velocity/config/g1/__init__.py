# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# Reuse agent configs from the stable task package.
from isaaclab_tasks.core.velocity.config.g1 import agents

# Warp tasks reuse the stable env cfgs; only physics (presets=newton_mjwarp)
# and MDP twins are swapped at construction (see adapt_cfg_for_warp).
_stable_pkg = agents.__name__.rsplit(".", 1)[0]

##
# Register Gym environments.
##

# gym.register(
#     id="Isaac-Velocity-Rough-G1-Warp-v0",
#     entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{_stable_pkg}.rough_env_cfg:G1RoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )


# gym.register(
#     id="Isaac-Velocity-Rough-G1-Warp-Play-v0",
#     entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{_stable_pkg}.rough_env_cfg:G1RoughEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
#     },
# )

gym.register(
    id="Isaac-Velocity-Flat-G1-Warp-v0",
    entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_stable_pkg}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-Warp-Play-v0",
    entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_stable_pkg}.flat_env_cfg:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
