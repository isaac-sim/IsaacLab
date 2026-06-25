# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless OVRTX visual-randomization backends (e.g. color via ``Renderer.write_attribute``)."""

from .ovrtx_visual_color_writer import OVRTXVisualColorWriter

__all__ = ["OVRTXVisualColorWriter"]
