# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers.

This module is a passthrough to `isaaclab.managers.manager_term_cfg` except for
`RewardTermCfg`, which is overridden for the Warp-based reward manager.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg as _ManagerTermBaseCfg
from isaaclab.managers.manager_term_cfg import *  # noqa: F401,F403
from isaaclab.utils import configclass


@configclass
class RewardTermCfg(_ManagerTermBaseCfg):
    """Configuration for a reward term.

    The function is expected to write the (unweighted) reward values into a
    pre-allocated Warp buffer provided by the manager.

    Expected signature:

    - ``func(env, out, **params) -> None``

    where ``out`` is a Warp array of shape ``(num_envs,)`` with float32 dtype.
    """

    func: Callable[..., None] = MISSING
    """The function to be called to fill the pre-allocated reward buffer."""

    weight: float = MISSING
    """The weight of the reward term."""
