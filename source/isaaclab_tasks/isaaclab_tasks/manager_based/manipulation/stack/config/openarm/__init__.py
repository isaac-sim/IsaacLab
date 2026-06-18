# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from . import stack_ik_abs_env_cfg, stack_joint_pos_env_cfg

gym.register(
    id="Isaac-Stack-Cube-OpenArm-IK-Abs-v0", 
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stack_ik_abs_env_cfg.OpenarmCubeStackEnvCfg,
    },
)