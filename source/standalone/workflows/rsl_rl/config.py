# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for parsing rsl-rl configuration files."""


import os
import yaml

from omni.isaac.orbit_envs import ORBIT_ENVS_DATA_DIR

__all__ = ["RSLRL_PPO_CONFIG_FILE", "parse_rslrl_cfg"]


RSLRL_PPO_CONFIG_FILE = {
    # classic
    "Isaac-Cartpole-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "rsl_rl/cartpole_ppo.yaml"),
    # manipulation
    "Isaac-Lift-Franka-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "rsl_rl/lift_ppo.yaml"),
    "Isaac-Reach-Franka-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "rsl_rl/reach_ppo.yaml"),
    # locomotion
    "Isaac-Velocity-Anymal-C-v0": os.path.join(ORBIT_ENVS_DATA_DIR, "rsl_rl/anymal_ppo.yaml"),
}
"""Mapping from environment names to PPO agent files."""


def parse_rslrl_cfg(task_name) -> dict:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name (str): The name of the environment.

    Returns:
        dict: A dictionary containing the parsed configuration.
    """
    # retrieve the default environment config file
    try:
        config_file = RSLRL_PPO_CONFIG_FILE[task_name]
    except KeyError:
        raise ValueError(f"Task not found: {task_name}")

    # parse agent configuration
    with open(config_file, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg
