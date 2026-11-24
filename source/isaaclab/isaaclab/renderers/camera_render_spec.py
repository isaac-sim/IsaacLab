# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated alias of :class:`~isaaclab.renderers.sensor_render_spec.SensorRenderSpec`.

``CameraRenderSpec`` is :class:`SensorRenderSpec`. Pass sensor prims as ``prim_paths``.
``camera_prim_paths`` still reads and sets that field.
"""

from __future__ import annotations

from typing import TypeAlias

from .sensor_render_spec import SensorRenderSpec

CameraRenderSpec: TypeAlias = SensorRenderSpec
