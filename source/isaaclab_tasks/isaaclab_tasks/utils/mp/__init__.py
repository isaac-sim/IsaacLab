# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Torch-first movement primitive utilities for Isaac Lab tasks.

This module provides the minimal interfaces and helpers needed to build MP-based
pipelines on top of IsaacLab environments without depending on fancy_gym.
"""

from .black_box_wrapper import BlackBoxWrapper
from .context_observation import ContextObsWrapper
from .factories import MP_DEFAULTS, get_basis_generator, get_controller, get_phase_generator, get_trajectory_generator
from .functional_wrapper import FunctionalMPWrapper
from .raw_interface import RawMPInterface
from .registry import make_mp_env, upgrade

__all__ = [
    "BlackBoxWrapper",
    "ContextObsWrapper",
    "MP_DEFAULTS",
    "RawMPInterface",
    "get_basis_generator",
    "get_controller",
    "get_phase_generator",
    "get_trajectory_generator",
    "FunctionalMPWrapper",
    "make_mp_env",
    "upgrade",
]
