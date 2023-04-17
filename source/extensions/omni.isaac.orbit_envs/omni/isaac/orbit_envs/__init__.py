# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing environments with OpenAI Gym interface.


We use OpenAI Gym registry to register the environment and their default configuration file.
The default configuration file is passed to the argument "kwargs" in the Gym specification registry.
The string is parsed into respective configuration container which needs to be passed to the environment
class. This is done using the function :meth:`load_default_env_cfg` in the sub-module
:mod:`omni.isaac.orbit.utils.parse_cfg`.

Note:
    This is a slight abuse of kwargs since they are meant to be directly passed into the environment class.
    Instead, we remove the key :obj:`cfg_file` from the "kwargs" dictionary and the user needs to provide
    the kwarg argument :obj:`cfg` while creating the environment.

Usage:
    >>> import gym
    >>> import omni.isaac.orbit_envs
    >>> from omni.isaac.orbit_envs.utils.parse_cfg import load_default_env_cfg
    >>>
    >>> task_name = "Isaac-Cartpole-v0"
    >>> cfg = load_default_env_cfg(task_name)
    >>> env = gym.make(task_name, cfg=cfg)
"""


import gym
import os
import toml

# Conveniences to other module directories via relative paths
ORBIT_ENVS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

ORBIT_ENVS_DATA_DIR = os.path.join(ORBIT_ENVS_EXT_DIR, "data")
"""Path to the extension data directory."""

ORBIT_ENVS_METADATA = toml.load(os.path.join(ORBIT_ENVS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ORBIT_ENVS_METADATA["package"]["version"]

##
# Classic control
##

gym.register(
    id="Isaac-Cartpole-v0",
    entry_point="omni.isaac.orbit_envs.classic.cartpole:CartpoleEnv",
    kwargs={"cfg_entry_point": "omni.isaac.orbit_envs.classic.cartpole:cartpole_cfg.yaml"},
)

gym.register(
    id="Isaac-Ant-v0",
    entry_point="omni.isaac.orbit_envs.classic.ant:AntEnv",
    kwargs={"cfg_entry_point": "omni.isaac.orbit_envs.classic.ant:ant_cfg.yaml"},
)

gym.register(
    id="Isaac-Humanoid-v0",
    entry_point="omni.isaac.orbit_envs.classic.humanoid:HumanoidEnv",
    kwargs={"cfg_entry_point": "omni.isaac.orbit_envs.classic.humanoid:humanoid_cfg.yaml"},
)

##
# Locomotion
##

gym.register(
    id="Isaac-Velocity-Anymal-C-v0",
    entry_point="omni.isaac.orbit_envs.locomotion.velocity:VelocityEnv",
    kwargs={"cfg_entry_point": "omni.isaac.orbit_envs.locomotion.velocity:VelocityEnvCfg"},
)

##
# Manipulation
##

gym.register(
    id="Isaac-Reach-Franka-v0",
    entry_point="omni.isaac.orbit_envs.manipulation.reach:ReachEnv",
    kwargs={"cfg_entry_point": "omni.isaac.orbit_envs.manipulation.reach:ReachEnvCfg"},
)

gym.register(
    id="Isaac-Lift-Franka-v0",
    entry_point="omni.isaac.orbit_envs.manipulation.lift:LiftEnv",
    kwargs={"cfg_entry_point": "omni.isaac.orbit_envs.manipulation.lift:LiftEnvCfg"},
)
