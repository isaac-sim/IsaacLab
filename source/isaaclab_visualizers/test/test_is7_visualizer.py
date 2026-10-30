# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Isaac Sim 7 viewer visualizer configuration and camera conversion."""

from __future__ import annotations

import math

import pytest
from isaaclab_visualizers.is7 import IS7VisualizerCfg
from isaaclab_visualizers.is7.is7_visualizer import _orbit_from_eye_lookat

from isaaclab.scene_data.scene_data_provider import REQUIRES_STAGE_AND_MODEL
from isaaclab.visualizers.visualizer_cfg import VISUALIZER_TYPES


def test_is7_is_registered():
    assert "is7" in VISUALIZER_TYPES
    assert REQUIRES_STAGE_AND_MODEL["is7"] == (True, False)
    assert IS7VisualizerCfg().visualizer_type == "is7"


@pytest.mark.parametrize(
    ("eye", "lookat", "distance", "azimuth", "elevation"),
    [
        ((4.0, 0.0, 0.0), (0.0, 0.0, 0.0), 4.0, 0.0, 0.0),
        ((0.0, -2.0, 0.0), (0.0, 0.0, 0.0), 2.0, -90.0, 0.0),
        ((2.0, 1.0, 1.0), (1.0, 1.0, 0.0), math.sqrt(2.0), 0.0, 45.0),
    ],
)
def test_orbit_from_eye_lookat(eye, lookat, distance, azimuth, elevation):
    orbit = _orbit_from_eye_lookat(eye, lookat)
    assert orbit["target"] == lookat
    assert orbit["distance"] == pytest.approx(distance)
    assert orbit["azimuth_deg"] == pytest.approx(azimuth, abs=1e-6)
    assert orbit["elevation_deg"] == pytest.approx(elevation)
