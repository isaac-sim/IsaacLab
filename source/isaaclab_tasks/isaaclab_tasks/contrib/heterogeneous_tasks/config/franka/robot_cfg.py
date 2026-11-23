# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot of the heterogeneous tasks: a Franka arm with a Robotiq 2F-85 gripper."""

from isaaclab.utils import clone

from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG

FRANKA_ROBOTIQ_CFG = clone(FRANKA_ROBOTIQ_GRIPPER_CFG)
FRANKA_ROBOTIQ_CFG.prim_path = "{ENV_REGEX_NS}/robot"
"""Franka arm with a Robotiq 2F-85 gripper, at the environment origin."""
