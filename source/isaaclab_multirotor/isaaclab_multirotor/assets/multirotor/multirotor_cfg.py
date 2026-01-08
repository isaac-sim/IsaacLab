# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_multirotor.actuators import ThrusterCfg

from .multirotor import Multirotor


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
        """RPS (Rotations Per Second) of the thrusters. Defaults to 100.0 (1/s) RPM for all thrusters."""

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

    rotor_directions: Sequence[int] | None = None
    """Sequence of rotor directions, -1 for clockwise, 1 for counter-clockwise. Length must match the number of thrusters."""
