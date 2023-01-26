# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for file I/O with yaml."""

import os
import yaml
from typing import Dict, Union

from omni.isaac.orbit.utils import class_to_dict


def load_yaml(filename: str) -> Dict:
    """Loads an input PKL file safely.

    Args:
        filename (str): The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        Dict: The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename) as f:
        data = yaml.full_load(f)
    return data


def dump_yaml(filename: str, data: Union[Dict, object]):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename (str): The path to save the file at.
        data (Union[Dict, object]): The data to save either a dictionary or class object.
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
        yaml.dump(data, f, default_flow_style=None)
