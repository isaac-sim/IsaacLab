# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_contrib.actuators import ThrusterCfg

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
        """Revolutions per second in [1/s] of the thrusters. Defaults to 100.0 for all thrusters."""

    # multirotor-specific configuration
    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the multirotor object."""

    actuators: dict[str, ThrusterCfg] = MISSING
    """Thruster actuators for the multirotor with corresponding thruster names."""

    # multirotor force application settings
    thruster_force_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Default force direction in body-local frame for thrusters. Defaults to Z-axis (upward)."""

    allocation_matrix: Sequence[Sequence[float]] | None = None
    """allocation matrix for control allocation"""

    rotor_directions: Sequence[int] | None = None
    """Sequence of rotor directions, -1 for clockwise, 1 for counter-clockwise."""

    def __post_init__(self):
        """Post initialization validation."""
        # Skip validation if actuators is MISSING
        if self.actuators is MISSING:
            return

        # Count the total number of thrusters from all actuator configs
        num_thrusters = 0
        for thruster_cfg in self.actuators.values():
            if hasattr(thruster_cfg, "thruster_names_expr") and thruster_cfg.thruster_names_expr is not None:
                num_thrusters += len(thruster_cfg.thruster_names_expr)

        # Validate rotor_directions matches number of thrusters
        if self.rotor_directions is not None:
            num_rotor_directions = len(self.rotor_directions)
            if num_thrusters != num_rotor_directions:
                raise ValueError(
                    f"Mismatch between number of thrusters ({num_thrusters}) and "
                    f"rotor_directions ({num_rotor_directions}). "
                    "They must have the same number of elements."
                )

        # Validate rps explicit entries match number of thrusters
        # Only validate if rps has explicit entries (not just a wildcard pattern)
        if hasattr(self.init_state, "rps") and self.init_state.rps is not None:
            rps_keys = list(self.init_state.rps.keys())
            # Check if rps uses a wildcard pattern (single key that's a regex)
            is_wildcard = len(rps_keys) == 1 and (rps_keys[0] == ".*" or rps_keys[0] == ".*:.*")

            if not is_wildcard and len(rps_keys) != num_thrusters:
                raise ValueError(
                    f"Mismatch between number of thrusters ({num_thrusters}) and "
                    f"rps entries ({len(rps_keys)}). "
                    "They must have the same number of elements when using explicit rps keys."
                )

        # Validate allocation_matrix second dimension matches number of thrusters
        if self.allocation_matrix is not None:
            if len(self.allocation_matrix) == 0:
                raise ValueError("Allocation matrix cannot be empty.")

            # Check that all rows have the same length
            num_cols = len(self.allocation_matrix[0])
            for i, row in enumerate(self.allocation_matrix):
                if len(row) != num_cols:
                    raise ValueError(
                        f"Allocation matrix row {i} has length {len(row)}, "
                        f"but expected {num_cols} (all rows must have the same length)."
                    )

            # Validate that the second dimension (columns) matches number of thrusters
            if num_cols != num_thrusters:
                raise ValueError(
                    f"Mismatch between number of thrusters ({num_thrusters}) and "
                    f"allocation matrix columns ({num_cols}). "
                    "The second dimension of the allocation matrix must match the number of thrusters."
                )
