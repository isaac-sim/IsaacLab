# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Solvers module for position-based dynamics simulations.

This module contains implementations of various constraint solvers,
including the Direct Position-Based Solver for Stiff Rods based on
Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods".
"""

from .rod_solver import *  # noqa: F401, F403

