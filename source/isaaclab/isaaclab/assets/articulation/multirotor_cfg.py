# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

import torch

from isaaclab.actuators import ThrusterCfg
from isaaclab.utils import configclass

from .articulation_cfg import ArticulationCfg
from .multirotor import Multirotor
from typing import Literal

@configclass
class MultirotorCfg(ArticulationCfg):
    """Configuration parameters for a multirotor articulation. 
    This extends the base articulation configuration to support multirotor-specific
    settings.
    """

    class_type: type = Multirotor

    @configclass
    class InitialStateCfg(ArticulationCfg.InitialStateCfg):
        """Initial state of the multirotor articulation."""

        # multirotor-specific initial state
        rps: dict[str, float] = {".*": 100.0}
        """RPS of the thrusters. Defaults to 100.0 for all thrusters."""

    # multirotor-specific configuration
    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the multirotor object."""

    actuators: dict[str, ThrusterCfg] = MISSING
    """Thruster actuators for the multirotor with corresponding thruster names."""

    # multirotor force application settings
    thruster_force_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Default force direction in body-local frame for thrusters. Defaults to Z-axis (upward)."""

    allocation_matrix: list[list[float]] | None = None
    """allocation matrix for control allocation"""
    
    rotor_directions: list[int] | None = None
    """List of rotor directions, -1 for clockwise, 1 for counter-clockwise."""
    
    force_application_mode: Literal["individual", "combined"] = "individual"
    """Force application mode: 'individual' applies forces at each thruster location, 
    'combined' applies combined wrench to base link. Defaults to 'individual'."""
