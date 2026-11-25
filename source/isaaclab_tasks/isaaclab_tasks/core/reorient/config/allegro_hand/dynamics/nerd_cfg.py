# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural model definition for AllegroHand."""

from isaaclab_neural import NeRDCfg

from isaaclab.utils import configclass


@configclass
class AllegroHandNeRDCfg(NeRDCfg):
    """Model architecture and inputs, shared by training and deployment."""

    backbone_spans: tuple[int, ...] = (1, 1, 1, 1)
    prediction: str = "newton_neural.predictions:PredictionFreeMotion"
