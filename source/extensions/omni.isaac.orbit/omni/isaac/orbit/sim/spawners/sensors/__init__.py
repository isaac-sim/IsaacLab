# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-module contains functions for spawning sensors in the simulation.

Currently, the following sensors are supported:

* Camera: A USD camera prim with settings for pinhole or fisheye projections.

"""

from __future__ import annotations

from .sensors import spawn_camera
from .sensors_cfg import FisheyeCameraCfg, PinholeCameraCfg

__all__ = [
    # camera
    "spawn_camera",
    "PinholeCameraCfg",
    "FisheyeCameraCfg",
]
