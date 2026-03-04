# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RMP-Flow controller."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RmpFlowControllerCfg:
    """Configuration for RMP-Flow controller (provided through LULA library)."""

    name: str = "rmp_flow"
    """Name of the controller. Supported: "rmp_flow", "rmp_flow_smoothed". Defaults to "rmp_flow"."""
    config_file: str = MISSING
    """Path to the configuration file for the controller."""
    urdf_file: str = MISSING
    """Path to the URDF model of the robot."""
    collision_file: str = MISSING
    """Path to collision model description of the robot."""
    frame_name: str = MISSING
    """Name of the robot frame for task space (must be present in the URDF)."""
    evaluations_per_frame: float = MISSING
    """Number of substeps during Euler integration inside LULA world model."""
    ignore_robot_state_updates: bool = False
    """If true, then state of the world model inside controller is rolled out. Defaults to False."""
