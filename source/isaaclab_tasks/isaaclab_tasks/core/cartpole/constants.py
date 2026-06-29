# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from isaaclab.utils.math import quat_from_euler_xyz

CARTPOLE_DISTANT_LIGHT_INTENSITY: float = 2000.0
CARTPOLE_DISTANT_LIGHT_COLOR: tuple[float, float, float] = (1.0, 1.0, 1.0)
CARTPOLE_DISTANT_LIGHT_ORIENTATION: tuple[float, float, float, float] = tuple(
    quat_from_euler_xyz(
        torch.tensor([0.0]),
        torch.tensor([math.radians(-45.0)]),
        torch.tensor([math.radians(-45.0)]),
    )[0].tolist()
)
