# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg
from . import joint_pos_env_cfg_rgb_rsl_rl

##
# Register Gym environments.
##

##
# Joint Position Control (rsl_rl)
##

gym.register(
    id="Isaac-Reach-Cube-Franka-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaCubeReachEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ReachCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-Cube-Franka-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaCubeReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ReachCubePPORunnerCfg",
    },
    disable_env_checker=True,
)


##
# RGB input based Control (rsl_rl)
##

gym.register(
    id="Isaac-Reach-Cube-Franka-v0-RGB-rsl_rl",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg_rgb_rsl_rl.FrankaCubeReachEnvCfg_rsl_rl,
        #"env_cfg_entry_point": ik_rel_env_cfg_rgb_rsl_rl.FrankaCubeReachEnvCfg_rsl_rl,
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_rgb:ReachCubePPORunnerCfg",
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_rgb_and_states:ReachCubePPORunnerCfg",
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_rgb_and_all_states:ReachCubePPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_rgbd:ReachCubePPORunnerCfg",
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_rgb_and_seg:ReachCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-Cube-Franka-Play-v0-RGB-rsl_rl",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg_rgb_rsl_rl.FrankaCubeReachEnvCfg_rsl_rl_PLAY,
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_rgb:ReachCubePPORunnerCfg",
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_rgb_and_states:ReachCubePPORunnerCfg",
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_rgb_and_all_states:ReachCubePPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_rgbd:ReachCubePPORunnerCfg",
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_rgb_and_seg:ReachCubePPORunnerCfg",
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Reach-Cube-Franka-IK-Abs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.FrankaCubeReachEnvCfg,
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Reach-Cube-Franka-IK-Rel-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaCubeReachEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)
