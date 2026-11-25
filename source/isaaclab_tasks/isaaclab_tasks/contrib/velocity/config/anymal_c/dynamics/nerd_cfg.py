# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural model definition for AnymalC."""

from isaaclab_neural import NeRDCfg

from isaaclab.utils import configclass


@configclass
class AnymalCNeRDCfg(NeRDCfg):
    """Model architecture and inputs, shared by training and deployment."""

    contact_bodies: str = "*FOOT"
