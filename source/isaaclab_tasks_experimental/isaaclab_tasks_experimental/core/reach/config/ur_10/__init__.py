# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# UR10 env disabled: USD asset has composition errors (broken asset file).
# Fails on both torch baseline and warp with:
#   RuntimeError: USD stage has composition errors while loading provided stage
# Re-enable once the UR10 USD asset is fixed.

# import gymnasium as gym
# from isaaclab_tasks.core.reach.config.ur_10 import agents

# Warp tasks reuse the stable env cfgs; only physics (presets=newton_mjwarp)
# and MDP twins are swapped at construction (see adapt_cfg_for_warp).
_stable_pkg = agents.__name__.rsplit(".", 1)[0]

# gym.register(
#     id="Isaac-Reach-UR10-Warp-v0",
#     entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{_stable_pkg}.joint_pos_env_cfg:UR10ReachEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10ReachPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Reach-UR10-Warp-Play-v0",
#     entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{_stable_pkg}.joint_pos_env_cfg:UR10ReachEnvCfg_PLAY",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10ReachPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )
