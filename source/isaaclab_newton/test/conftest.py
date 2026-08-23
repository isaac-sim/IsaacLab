# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for the isaaclab_newton test suite."""

import warp as wp

# Newton tests do not exercise Warp autodiff.  Set this before pytest imports test
# modules so kernels are compiled without backward support, matching the training,
# play, and benchmark entry points.
wp.config.enable_backward = False
