# Copyright (c) 2022-2023, The ORBIT Project Developers.
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
    >>> import omni.isaac.orbit_tasks
    >>> from omni.isaac.orbit_tasks.utils.parse_cfg import load_default_env_cfg
    >>>
    >>> task_name = "Isaac-Cartpole-v0"
    >>> cfg = load_default_env_cfg(task_name)
    >>> env = gym.make(task_name, cfg=cfg)
"""

from __future__ import annotations

import os
import toml

# Conveniences to other module directories via relative paths
ORBIT_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

ORBIT_TASKS_METADATA = toml.load(os.path.join(ORBIT_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ORBIT_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##

from .utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["locomotion.velocity.config.anymal_d", "classic", "manipulation", "utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
