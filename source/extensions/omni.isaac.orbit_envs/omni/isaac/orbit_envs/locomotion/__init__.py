# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion environments for legged robots.

These environments are based on the `legged_gym` environments provided by Rudin et al.

Reference:
    https://github.com/leggedrobotics/legged_gym
"""

import gym

from .locomotion_env_cfg import LocomotionEnvRoughCfg, LocomotionEnvRoughCfg_PLAY

__all__ = ["LocomotionEnvRoughCfg", "LocomotionEnvRoughCfg_PLAY"]

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-v0",
    entry_point="omni.isaac.orbit.envs.rl_env:RLEnv",
    kwargs={"cfg_entry_point": "omni.isaac.orbit_envs.locomotion:LocomotionEnvRoughCfg"},
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-Play-v0",
    entry_point="omni.isaac.orbit.envs.rl_env:RLEnv",
    kwargs={"cfg_entry_point": "omni.isaac.orbit_envs.locomotion:LocomotionEnvRoughCfg_PLAY"},
)
