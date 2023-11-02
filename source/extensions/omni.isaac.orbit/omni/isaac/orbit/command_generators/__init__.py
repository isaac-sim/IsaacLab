# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This submodule provides command generators for goal-conditioned tasks.

The command generators are used to generate commands for the agent to execute. The command generators act
as utility classes to make it convenient to switch between different command generation strategies within
the same environment. For instance, in an environment consisting of a quadrupedal robot, the command to it
could be a velocity command or position command. By keeping the command generation logic separate from the
environment, it is easy to switch between different command generation strategies.

The command generators are implemented as classes that inherit from the :class:`CommandGeneratorBase` class.
Each command generator class should also have a corresponding configuration class that inherits from the
:class:`CommandGeneratorBaseCfg` class.
"""

from .command_generator_base import CommandGeneratorBase
from .command_generator_cfg import (
    CommandGeneratorBaseCfg,
    NormalVelocityCommandGeneratorCfg,
    NullCommandGeneratorCfg,
    TerrainBasedPositionCommandGeneratorCfg,
    UniformVelocityCommandGeneratorCfg,
)
from .null_command_generator import NullCommandGenerator
from .position_command_generator import TerrainBasedPositionCommandGenerator
from .velocity_command_generator import NormalVelocityCommandGenerator, UniformVelocityCommandGenerator

__all__ = [
    "CommandGeneratorBase",
    "CommandGeneratorBaseCfg",
    # null command generator
    "NullCommandGenerator",
    "NullCommandGeneratorCfg",
    # velocity command generators
    "UniformVelocityCommandGenerator",
    "UniformVelocityCommandGeneratorCfg",
    "NormalVelocityCommandGenerator",
    "NormalVelocityCommandGeneratorCfg",
    # position command generators
    "TerrainBasedPositionCommandGenerator",
    "TerrainBasedPositionCommandGeneratorCfg",
]
