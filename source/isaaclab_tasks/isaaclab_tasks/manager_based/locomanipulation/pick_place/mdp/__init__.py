# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that are specific to the locomanipulation environments."""

import isaaclab.envs.mdp as _parent_mdp

from .actions import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403


def __getattr__(name):
    return getattr(_parent_mdp, name)
