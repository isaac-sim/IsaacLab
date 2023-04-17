# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for parsing sb3 configuration files."""

import os
import yaml
from torch import nn as nn  # noqa: F401

from omni.isaac.orbit_envs import ORBIT_ENVS_DATA_DIR

__all__ = ["SB3_PPO_CONFIG_FILE", "parse_sb3_cfg"]

SB3_PPO_CONFIG_FILE = {
    # classic
    "Isaac-Cartpole-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "sb3/cartpole_ppo.yaml"),
    "Isaac-Ant-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "sb3/ant_ppo.yaml"),
    "Isaac-Humanoid-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "sb3/humanoid_ppo.yaml"),
    # manipulation
    "Isaac-Reach-Franka-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "sb3/reach_ppo.yaml"),
}
"""Mapping from environment names to PPO agent files."""


def parse_sb3_cfg(task_name) -> dict:
    """Parse configuration for Stable-baselines3 agent based on inputs.

    Args:
        task_name (str): The name of the environment.

    Returns:
        dict: A dictionary containing the parsed configuration.
    """
    # retrieve the default environment config file
    try:
        config_file = SB3_PPO_CONFIG_FILE[task_name]
    except KeyError:
        raise ValueError(f"Task not found: {task_name}. Configurations exist for {SB3_PPO_CONFIG_FILE.keys()}")

    # parse agent configuration
    with open(config_file, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # check config is valid
    if cfg is None:
        raise ValueError(f"Config file is empty: {config_file}")

    # post-process certain arguments
    # reference: https://github.com/DLR-RM/rl-baselines3-zoo/blob/0e5eb145faefa33e7d79c7f8c179788574b20da5/utils/exp_manager.py#L358
    for kwargs_key in ["policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"]:
        if kwargs_key in cfg and isinstance(cfg[kwargs_key], str):
            cfg[kwargs_key] = eval(cfg[kwargs_key])

    return cfg
