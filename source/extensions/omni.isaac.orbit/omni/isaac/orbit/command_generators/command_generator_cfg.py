# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import ClassVar, Tuple

from omni.isaac.orbit.utils import configclass

"""
Base command generator.
"""


@configclass
class CommandGeneratorBaseCfg:
    """Configuration for the base command generator."""

    class_name: ClassVar[str] = MISSING
    """Name of the command generator class."""
    resampling_time_range: Tuple[float, float] = MISSING
    """Time before commands are changed [s]."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""


"""
Locomotion-specific command generators.
"""


@configclass
class UniformVelocityCommandGeneratorCfg(CommandGeneratorBaseCfg):
    """Configuration for the uniform velocity command generator."""

    class_name = "UniformVelocityCommandGenerator"

    robot_attr: str = MISSING
    """Name of the robot attribute from the environment."""
    heading_command: bool = MISSING
    """Whether to use heading command or angular velocity command.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """
    rel_standing_envs: float = MISSING
    """Probability threshold for environments where the robots that are standing still."""
    rel_heading_envs: float = MISSING
    """Probability threshold for environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command)."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: Tuple[float, float] = MISSING  # min max [m/s]
        lin_vel_y: Tuple[float, float] = MISSING  # min max [m/s]
        ang_vel_z: Tuple[float, float] = MISSING  # min max [rad/s]
        heading: Tuple[float, float] = MISSING  # [rad]

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class NormalVelocityCommandGeneratorCfg(UniformVelocityCommandGeneratorCfg):
    """Configuration for the normal velocity command generator."""

    class_name = "NormalVelocityCommandGenerator"
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.

    @configclass
    class Ranges:
        """Normal distribution ranges for the velocity commands."""

        mean_vel: Tuple[float, float, float] = MISSING
        """Mean velocity for the normal distribution.

        The tuple contains the mean linear-x, linear-y, and angular-z velocity.
        """
        std_vel: Tuple[float, float, float] = MISSING
        """Standard deviation for the normal distribution.

        The tuple contains the standard deviation linear-x, linear-y, and angular-z velocity.
        """
        zero_prob: Tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class TerrainBasedPositionCommandGeneratorCfg(CommandGeneratorBaseCfg):
    """Configuration for the terrain-based position command generator."""

    class_name = "TerrainBasedPositionCommandGenerator"

    robot_attr: str = MISSING
    """Name of the robot attribute from the environment."""
    rel_standing_envs: float = MISSING
    """Probability threshold for environments where the robots that are standing still."""
    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        heading: Tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""
