# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# UR10 env disabled: USD asset has composition errors (broken asset file).
# Fails on both torch baseline and warp with:
#   RuntimeError: USD stage has composition errors while loading provided stage
# Re-enable once the UR10 USD asset is fixed.

# import gymnasium as gym
# from isaaclab_tasks.manager_based.manipulation.reach.config.ur_10 import agents

# gym.register(
#     id="Isaac-Reach-UR10-Warp-v0",
#     entry_point="isaaclab_experimental.envs:ManagerBasedRLEnvWarp",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10ReachEnvCfg",
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
#         "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10ReachEnvCfg_PLAY",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10ReachPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )
