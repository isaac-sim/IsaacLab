# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    agents,
    stack_ik_abs_env_cfg,
    stack_ik_rel_blueprint_env_cfg,
    stack_ik_rel_env_cfg,
    stack_ik_rel_instance_randomize_env_cfg,
    stack_ik_rel_visuomotor_cosmos_env_cfg,
    stack_ik_rel_visuomotor_env_cfg,
    stack_joint_pos_env_cfg,
    stack_joint_pos_instance_randomize_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Stack-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Instance-Randomize-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_instance_randomize_env_cfg.FrankaCubeStackInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_env_cfg.FrankaCubeStackEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_visuomotor_env_cfg.FrankaCubeStackVisuomotorEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_visuomotor_cosmos_env_cfg.FrankaCubeStackVisuomotorCosmosEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_cosmos.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_abs_env_cfg.FrankaCubeStackEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_instance_randomize_env_cfg.FrankaCubeStackInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_blueprint_env_cfg.FrankaCubeStackBlueprintEnvCfg,
    },
    disable_env_checker=True,
)
