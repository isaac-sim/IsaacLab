# Copyright [2023] Boston Dynamics AI Institute, Inc.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-module contains spawners that spawn USD-based light prims.

There are various different kinds of lights that can be spawned into the USD stage.
Please check the Omniverse documentation for `lighting overview
<https://docs.omniverse.nvidia.com/materials-and-rendering/latest/103/lighting.html>`_.
"""

from __future__ import annotations

from .lights import spawn_light
from .lights_cfg import CylinderLightCfg, DiskLightCfg, DistantLightCfg, DomeLightCfg, LightCfg, SphereLightCfg

__all__ = [
    # base class
    "LightCfg",
    "spawn_light",
    # derived classes
    "CylinderLightCfg",
    "DiskLightCfg",
    "DistantLightCfg",
    "DomeLightCfg",
    "SphereLightCfg",
]
