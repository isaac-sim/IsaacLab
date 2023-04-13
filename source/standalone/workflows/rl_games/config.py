# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for parsing rl-games configuration files."""

import os
import yaml

from omni.isaac.orbit_envs import ORBIT_ENVS_DATA_DIR

__all__ = ["RLG_PPO_CONFIG_FILE", "parse_rlg_cfg"]


RLG_PPO_CONFIG_FILE = {
    # classic
    "Isaac-Cartpole-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "rl_games/cartpole_ppo.yaml"),
    "Isaac-Ant-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "rl_games/ant_ppo.yaml"),
    "Isaac-Humanoid-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "rl_games/humanoid_ppo.yaml"),
    # manipulation
    "Isaac-Lift-Franka-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "rl_games/lift_ppo.yaml"),
    "Isaac-Reach-Franka-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "rl_games/reach_ppo.yaml"),
}
"""Mapping from environment names to PPO agent files."""


def parse_rlg_cfg(task_name) -> dict:
    """Parse configuration based on command line arguments.

    Args:
        task_name (str): The name of the environment.

    Returns:
        dict: A dictionary containing the parsed configuration.
    """
    # retrieve the default environment config file
    try:
        config_file = RLG_PPO_CONFIG_FILE[task_name]
    except KeyError:
        raise ValueError(f"Task not found: {task_name}")

    # parse agent configuration
    with open(config_file, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    return cfg
