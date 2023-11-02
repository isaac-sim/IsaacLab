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
    >>> import omni.isaac.orbit_envs
    >>> from omni.isaac.orbit_envs.utils.parse_cfg import load_default_env_cfg
    >>>
    >>> task_name = "Isaac-Cartpole-v0"
    >>> cfg = load_default_env_cfg(task_name)
    >>> env = gym.make(task_name, cfg=cfg)
"""

from __future__ import annotations

import importlib
import os
import pkgutil
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
# Register Gym environments.
##


def _import_all(package_name: str, blacklist_pkgs: list[str] = None):
    """Import all sub-packages in a package recursively.

    It is easier to use this function to import all sub-packages in a package recursively
    than to manually import each sub-package.

    It replaces the need of the following code:

    .. code-block:: python

        import .locomotion.velocity
        import .manipulation.reach
        import .manipulation.lift

    Args:
        package_name: The package name.
        blacklist_pkgs: The list of blacklisted packages to skip. Defaults to None,
            which means no packages are blacklisted.
    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []
    # Import the package
    package = importlib.import_module(package_name)
    # Import all Python files
    for file_name, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        # check blacklisted
        if any([pkg_name in module_name for pkg_name in blacklist_pkgs]):
            continue
        if is_pkg:
            importlib.import_module(module_name)
            _import_all(module_name, blacklist_pkgs)


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["locomotion.velocity.config.anymal_d", "classic", "manipulation", "utils"]
# Import all configs in this package
_import_all(__name__, _BLACKLIST_PKGS)
