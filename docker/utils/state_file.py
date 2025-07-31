# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import configparser
from configparser import ConfigParser
from pathlib import Path
from typing import Any


class StateFile:
    """A class to manage state variables parsed from a configuration file.

    This class provides a simple interface to set, get, and delete variables from a configuration
    object. It also provides the ability to save the configuration object to a file.

    It thinly wraps around the ConfigParser class from the configparser module.
    """

    def __init__(self, path: Path, namespace: str | None = None):
        """Initialize the class instance and load the configuration file.

        Args:
            path: The path to the configuration file.
            namespace: The default namespace to use when setting and getting variables.
                Namespace corresponds to a section in the configuration file. Defaults to None,
                meaning  all member functions will have to specify the section explicitly,
                or :attr:`StateFile.namespace` must be set manually.
        """
        self.path = path
        self.namespace = namespace

        # load the configuration file
        self.load()

    def __del__(self):
        """
        Save the loaded configuration to the initial file path upon deconstruction. This helps
        ensure that the configuration file is always up to date.
        """
        # save the configuration file
        self.save()

    """
    Operations.
    """

    def set_variable(self, key: str, value: Any, section: str | None = None):
        """Set a variable into the configuration object.

        Note:
            Since we use the ConfigParser class, the section names are case-sensitive but the keys are not.

        Args:
            key: The key of the variable to be set.
            value: The value of the variable to be set.
            section: The section of the configuration object to set the variable in.
                Defaults to None, in which case the default section is used.

        Raises:
            configparser.Error: If no section is specified and the default section is None.
        """
        # resolve the section
        if section is None:
            if self.namespace is None:
                raise configparser.Error("No section specified. Please specify a section or set StateFile.namespace.")
            section = self.namespace

        # create section if it does not exist
        if section not in self.loaded_cfg.sections():
            self.loaded_cfg.add_section(section)
        # set the variable
        self.loaded_cfg.set(section, key, value)

    def get_variable(self, key: str, section: str | None = None) -> Any:
        """Get a variable from the configuration object.

        Note:
            Since we use the ConfigParser class, the section names are case-sensitive but the keys are not.

        Args:
            key: The key of the variable to be loaded.
            section: The section of the configuration object to read the variable from.
                Defaults to None, in which case the default section is used.

        Returns:
            The value of the variable. It is None if the key does not exist.

        Raises:
            configparser.Error: If no section is specified and the default section is None.
        """
        # resolve the section
        if section is None:
            if self.namespace is None:
                raise configparser.Error("No section specified. Please specify a section or set StateFile.namespace.")
            section = self.namespace

        return self.loaded_cfg.get(section, key, fallback=None)

    def delete_variable(self, key: str, section: str | None = None):
        """Delete a variable from the configuration object.

        Note:
            Since we use the ConfigParser class, the section names are case-sensitive but the keys are not.

        Args:
            key: The key of the variable to be deleted.
            section: The section of the configuration object to remove the variable from.
                Defaults to None, in which case the default section is used.

        Raises:
            configparser.Error: If no section is specified and the default section is None.
            configparser.NoSectionError: If the section does not exist in the configuration object.
            configparser.NoOptionError: If the key does not exist in the section.
        """
        # resolve the section
        if section is None:
            if self.namespace is None:
                raise configparser.Error("No section specified. Please specify a section or set StateFile.namespace.")
            section = self.namespace

        # check if the section exists
        if section not in self.loaded_cfg.sections():
            raise configparser.NoSectionError(f"Section '{section}' does not exist in the file: {self.path}")

        # check if the key exists
        if self.loaded_cfg.has_option(section, key):
            self.loaded_cfg.remove_option(section, key)
        else:
            raise configparser.NoOptionError(option=key, section=section)

    """
    Operations - File I/O.
    """

    def load(self):
        """Load the configuration file into memory.

        This function reads the contents of the configuration file into memory.
        If the file does not exist, it creates an empty file.
        """
        self.loaded_cfg = ConfigParser()
        self.loaded_cfg.read(self.path)

    def save(self):
        """Save the configuration file to disk."""
        with open(self.path, "w+") as f:
            self.loaded_cfg.write(f)
