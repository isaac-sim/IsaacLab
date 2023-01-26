# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for parsing robomimic configuration files."""


import os

from omni.isaac.orbit_envs import ORBIT_ENVS_DATA_DIR

ROBOMIMIC_CONFIG_FILES_DICT = {
    "Isaac-Lift-Franka-v0": {
        "bc": os.path.join(ORBIT_ENVS_DATA_DIR, "robomimic/lift_bc.json"),
        "bcq": os.path.join(ORBIT_ENVS_DATA_DIR, "robomimic/lift_bcq.json"),
    }
}
"""Mapping from environment names to imitation learning config files."""

__all__ = ["ROBOMIMIC_CONFIG_FILES_DICT"]
