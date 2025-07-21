# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn sensors in the simulation.

Currently, the following sensors are supported:

* Camera: A USD camera prim with settings for pinhole or fisheye projections.

"""

from .sensors import spawn_camera
from .sensors_cfg import FisheyeCameraCfg, PinholeCameraCfg
