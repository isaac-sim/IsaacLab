# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

##
# Deprecated inverse-kinematics task aliases.
##

gym.register(
    id="IsaacContrib-Reach-Franka-IK-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaReachEnvCfg",
        "deprecated": {"alias": "--task Isaac-Reach-Franka physics=isaacsim_physx presets=diffik_abs"},
    },
    disable_env_checker=True,
)
