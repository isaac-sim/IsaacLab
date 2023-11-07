# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing environments contributed by the community.


We use OpenAI Gym registry to register the environment and their default configuration file.
The default configuration file is passed to the argument "kwargs" in the Gym specification registry.
The string is parsed into respective configuration container which needs to be passed to the environment
class. This is done using the function :meth:`load_cfg_from_registry` in the sub-module
:mod:`omni.isaac.orbit.utils.parse_cfg`.

Note:
    This is a slight abuse of kwargs since they are meant to be directly passed into the environment class.
    Instead, we remove the key :obj:`cfg_file` from the "kwargs" dictionary and the user needs to provide
    the kwarg argument :obj:`cfg` while creating the environment.

Usage:
    >>> import gymnasium as gym
    >>> import omni.isaac.contrib_tasks
    >>> from omni.isaac.orbit_tasks.utils.parse_cfg import load_cfg_from_registry
    >>>
    >>> task_name = "Isaac-Contrib-<my-registered-env-name>-v0"
    >>> cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    >>> env = gym.make(task_name, cfg=cfg)
"""

from __future__ import annotations

import gymnasium as gym  # noqa: F401
import os
import toml

# Conveniences to other module directories via relative paths
ORBIT_CONTRIB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

ORBIT_CONTRIB_TASKS_METADATA = toml.load(os.path.join(ORBIT_CONTRIB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ORBIT_CONTRIB_TASKS_METADATA["package"]["version"]
