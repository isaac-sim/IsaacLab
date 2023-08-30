# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.orbit.utils import configclass

from .command_generator_base import CommandGeneratorBase
from .position_command_generator import TerrainBasedPositionCommandGenerator
from .velocity_command_generator import NormalVelocityCommandGenerator, UniformVelocityCommandGenerator

"""
Base command generator.
"""


@configclass
class CommandGeneratorBaseCfg:
    """Configuration for the base command generator."""

    class_name: type[CommandGeneratorBase] = MISSING
    """The command generator class to use."""
    resampling_time_range: tuple[float, float] = MISSING
    """Time before commands are changed [s]."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""


"""
Locomotion-specific command generators.
"""


@configclass
class UniformVelocityCommandGeneratorCfg(CommandGeneratorBaseCfg):
    """Configuration for the uniform velocity command generator."""

    class_name = UniformVelocityCommandGenerator

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
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

        lin_vel_x: tuple[float, float] = MISSING  # min max [m/s]
        lin_vel_y: tuple[float, float] = MISSING  # min max [m/s]
        ang_vel_z: tuple[float, float] = MISSING  # min max [rad/s]
        heading: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class NormalVelocityCommandGeneratorCfg(UniformVelocityCommandGeneratorCfg):
    """Configuration for the normal velocity command generator."""

    class_name = NormalVelocityCommandGenerator
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.

    @configclass
    class Ranges:
        """Normal distribution ranges for the velocity commands."""

        mean_vel: tuple[float, float, float] = MISSING
        """Mean velocity for the normal distribution.

        The tuple contains the mean linear-x, linear-y, and angular-z velocity.
        """
        std_vel: tuple[float, float, float] = MISSING
        """Standard deviation for the normal distribution.

        The tuple contains the standard deviation linear-x, linear-y, and angular-z velocity.
        """
        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class TerrainBasedPositionCommandGeneratorCfg(CommandGeneratorBaseCfg):
    """Configuration for the terrain-based position command generator."""

    class_name = TerrainBasedPositionCommandGenerator

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    rel_standing_envs: float = MISSING
    """Probability threshold for environments where the robots that are standing still."""
    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""
