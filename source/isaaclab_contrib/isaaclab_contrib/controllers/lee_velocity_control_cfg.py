# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from .lee_controller_base_cfg import LeeControllerBaseCfg


@configclass
class LeeVelControllerCfg(LeeControllerBaseCfg):
    """Configuration for a Lee-style geometric quadrotor velocity controller.

    Unless otherwise noted, vectors are ordered as (x, y, z) in the simulation world/body frames.
    When :attr:`randomize_params` is True, gains are sampled uniformly per environment between
    their corresponding ``*_min`` and ``*_max`` bounds at reset.
    """

    K_vel_range: tuple[tuple[float, float, float], tuple[float, float, float]] = MISSING
    """Velocity error proportional gain range about body axes [unitless].

    This is a tuple of two tuples containing the minimum and maximum gains for each axis (x, y, z).
    Format: ((min_x, min_y, min_z), (max_x, max_y, max_z))

    Example:
        ((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)) for ARL Robot 1
    """
