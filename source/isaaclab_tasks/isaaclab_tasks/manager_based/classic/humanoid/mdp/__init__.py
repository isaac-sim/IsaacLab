# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the humanoid environment."""

import isaaclab.envs.mdp as _parent_mdp

from .observations import *
from .rewards import *


def __getattr__(name):
    return getattr(_parent_mdp, name)
