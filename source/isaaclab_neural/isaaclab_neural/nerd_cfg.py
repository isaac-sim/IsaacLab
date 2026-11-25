# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Model definitions independent of recording, optimization and runtime solvers."""

from isaaclab.utils import configclass


@configclass
class NeRDCfg:
    """FlashNeRD architecture and physical inputs, persisted with its checkpoint.

    Recording reads :attr:`prediction` and :attr:`contact_bodies`, which select the signals a dataset holds.
    Fitting builds the model, so the architecture can change without recording again.
    """

    backbone_spans: tuple[int, ...] = (4, 4, 4, 4)
    """Temporal attention spans of the model backbone [frames]."""

    prediction: str = "newton_neural.predictions:PredictionStateIncrement"
    """Native prediction class, which decides the labels that a dataset records."""

    contact_bodies: str = ""
    """Body-name glob for ground probes, recorded as contact input; empty means no contact input."""
