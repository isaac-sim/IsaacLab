# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import configparser
from configparser import ConfigParser
from pathlib import Path
from typing import Any


def load_cfg_file(path: Path) -> ConfigParser:
    """
    Load the contents of a config file.

    If the file exists, read its contents and return them as a dictionary.
    If the file does not exist, create an empty file and return an empty dictionary.

    Args:
        statefile: The path to the config file.

    Returns:
        dict: The contents of the config file as a dictionary.
    """
    cfg = ConfigParser()
    cfg.read(path)
    return cfg


def save_cfg_file(path: Path, cfg: ConfigParser):
    """
    Save a dictionary to a config file.

    Args:
        path: The path to the config file.
        data: The data to be saved to the config file.
    """
    with open(path, "w+") as file:
        cfg.write(file)


class Statefile:
    """
    A class to manage state variables stored in a cfg file.
    """

    def __init__(self, path: Path, namespace: str | None = None):
        """
        Initialize the Statefile object with the path to the cfg file.

        Args:
            path: The path to the cfg file.
            namespace: Namespace a section of the cfg.
                Defaults to None, and all member functions will have
                to specify section or else set Statefile.namespace directly.
        """
        self.path = path
        self.namespace = namespace
        self.load_cfg()

    def __del__(self):
        """
        Save self.loaded_cfg to self.path upon deconstruction
        """
        self.save_cfg()

    def set_variable(self, key: str, value: Any, section: str | None = None):
        """
        Set a variable in the cfg file.

        Args:
            key: The key of the variable to be set.
            value: The value of the variable to be set.
            section: section of the cfg. Defaults to the self.namespace
        """
        if section is None:
            if self.namespace is None:
                raise configparser.Error("No section specified")
            section = self.namespace
        if section not in self.loaded_cfg.sections():
            self.loaded_cfg.add_section(section)
        self.loaded_cfg.set(section, key, value)

    def load_variable(self, key: str, section: str | None = None) -> Any:
        """
        Load a variable from the cfg file.

        Args:
            key: The key of the variable to be loaded.
            section: section of the cfg. Defaults to the self.namespace

        Returns:
            any: The value of the variable, or None if the key does not exist.
        """
        if section is None:
            if self.namespace is None:
                raise configparser.Error("No section specified")
            section = self.namespace
        return self.loaded_cfg.get(section, key, fallback=None)

    def delete_variable(self, key: str, section: str | None = None):
        """
        Delete a variable from the cfg file.

        Args:
            key: The key of the variable to be deleted.
            section: section of the cfg. Defaults to self.namespace
        """
        if section is None:
            if self.namespace is None:
                raise configparser.Error("No section specified")
            section = self.namespace
        if section not in self.loaded_cfg.sections():
            raise configparser.NoSectionError(f"Section {section} does not exist in {self.path}")
        if self.loaded_cfg.has_option(section, key):
            self.loaded_cfg.remove_option(section, key)
        else:
            raise configparser.NoOptionError(option=key, section=section)

    def load_cfg(self):
        """
        Calls load_cfg_file() to populate self.loaded_cfg with the
        data stored at self.path
        """
        self.loaded_cfg = load_cfg_file(self.path)

    def save_cfg(self):
        """
        Calls save_cfg_file() to write the contents of self.loaded_cfg
        to the file at self.path
        """
        save_cfg_file(self.path, self.loaded_cfg)
