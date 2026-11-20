# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation groups for active-gaze camera tasks."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab_fovi import mdp

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass


@configclass
class GazeObservationsCfg(ObservationGroupCfg):
    """The gaze, in field radii, and its velocity, in field radii per second."""

    state = ObservationTermCfg(func=mdp.gaze_state)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class OracleObservationsCfg(ObservationGroupCfg):
    """The teacher's gaze labels. Training labels, never a policy input."""

    labels: ObservationTermCfg = MISSING

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
