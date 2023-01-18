# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Submodule containing configuration instances for commonly used robots.
"""

from .anydrive import ANYDRIVE_3_ACTUATOR_CFG, ANYDRIVE_SIMPLE_ACTUATOR_CFG
from .franka import PANDA_HAND_MIMIC_GROUP_CFG

__all__ = [
    # ANYmal actuators
    "ANYDRIVE_SIMPLE_ACTUATOR_CFG",
    "ANYDRIVE_3_ACTUATOR_CFG",
    # Franka panda actuators
    "PANDA_HAND_MIMIC_GROUP_CFG",
]
