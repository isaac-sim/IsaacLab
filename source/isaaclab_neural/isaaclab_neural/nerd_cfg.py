# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Model definitions independent of recording, optimization and runtime solvers."""

from isaaclab.utils import configclass


@configclass
class NeRDCfg:
    """FlashNeRD architecture and physical inputs, persisted with its checkpoint."""

    backbone_spans: tuple[int, ...] = (4, 4, 4, 4)
    """Temporal attention spans of the model backbone [frames]."""

    prediction: str = "newton_neural.predictions:PredictionStateIncrement"
    """Native prediction class."""

    contact_bodies: str = ""
    """Body-name glob for ground probes; empty means no contact input."""
