# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Code adapted from https://github.com/leggedrobotics/nav-suite

# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .navigation_actions import NavigationSE2Action


@configclass
class NavigationSE2ActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = NavigationSE2Action
    """ Class of the action term."""

    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""

    low_level_action: ActionTermCfg | list[ActionTermCfg] = MISSING
    """Configuration of the low level action term."""

    low_level_policy_file: str = MISSING
    """Path to the low level policy file."""

    freeze_low_level_policy: bool = True
    """Whether to freeze the low level policy.

    Can improve performance but will also eliminate possible functions such as `reset`."""

    low_level_obs_group: str = "low_level_policy"
    """Observation group of the low level policy."""

    action_dim: int = 3
    """Dimension of the action space. Default is 3 [vx, vy, omega]."""

    clip_mode: Literal["none", "minmax", "tanh"] = "none"
    """Clip mode for the action space. Default is "none".

    "minmax": Clip the action space to the range of the min and max values defined in :attr:`clip`.
    "tanh": Clip the action space to the range of the tanh function, i.e., each action dimension between -1 and 1.
    "none": No clipping is applied.
    """

    clip: list[tuple[float, float]] | None = None
    """Clip the action space. Default is None.

    For each action dimension, provide a tuple of (min, max) values."""

    scale: float | list[float] | None = None
    """Scale the action space. Default is None.

    .. note::
        Scale is applied after clipping. If a list is provided, it must be of the same length as the number of action dimensions.
    """

    offset: float | list[float] | None = None
    """Offset the action space. Default is None.

    .. note::
        Offset is applied after scaling. If a list is provided, it must be of the same length as the number of action dimensions.
    """
