# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless Newton-Warp visual-randomization backends (e.g. color via ``model.shape_color``)."""

from .newton_shape_color_writer import NewtonShapeColorWriter

__all__ = ["NewtonShapeColorWriter"]
