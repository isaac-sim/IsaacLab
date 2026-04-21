# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for file I/O with yaml."""

import logging
import os

import yaml

from isaaclab.utils import class_to_dict


def load_yaml(filename: str) -> dict:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename) as f:
        data = yaml.full_load(f)
    return data


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.
    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # convert data into dictionary
    if not isinstance(data, dict):
        data = class_to_dict(data)
    # save data
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)


def dump_resolved_cfg(cfg: object, log_dir: str | None, logger: logging.Logger | None = None):
    """Dump a fully-resolved config to ``<log_dir>/params/resolved_env.yaml``.

    This is intended to be called at the end of environment ``__init__``
    after all ``__post_init__`` hooks, preset resolution, and training-script
    mutations have been applied, giving users a single source of truth for the
    values the environment actually uses.

    Args:
        cfg: The config object to serialize (typically ``self.cfg``).
        log_dir: The logging directory. When ``None``, the dump is silently
            skipped.
        logger: Optional logger for status messages. When ``None``, messages
            are suppressed.
    """
    if log_dir is None:
        return
    resolved_path = os.path.join(log_dir, "params", "resolved_env.yaml")
    try:
        dump_yaml(resolved_path, cfg)
        if logger is not None:
            logger.info("Resolved env config written to %s", resolved_path)
    except Exception:
        if logger is not None:
            logger.warning("Failed to dump resolved env config to %s", resolved_path, exc_info=True)
