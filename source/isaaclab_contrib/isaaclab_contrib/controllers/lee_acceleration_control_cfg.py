# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass

from .lee_controller_base_cfg import LeeControllerBaseCfg


@configclass
class LeeAccControllerCfg(LeeControllerBaseCfg):
    """Configuration for a Lee-style geometric quadrotor acceleration controller.

    Unless otherwise noted, vectors are ordered as (x, y, z) in the simulation world/body frames.
    When :attr:`randomize_params` is True, gains are sampled uniformly per environment between
    their corresponding ``*_min`` and ``*_max`` bounds at reset.
    """

    pass
