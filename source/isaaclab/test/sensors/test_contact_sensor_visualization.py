# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared contact-force visualization."""

import torch

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
from isaaclab.sensors.contact_sensor import visualization


class _FakeVisualizationMarkers:
    """Capture marker updates without creating a rendering backend."""

    last_values = None

    def __init__(self, cfg: VisualizationMarkersCfg):
        self.cfg = cfg

    def set_visibility(self, visible: bool) -> None:
        pass

    def visualize(self, positions: torch.Tensor, orientations: torch.Tensor, scales: torch.Tensor) -> None:
        type(self).last_values = (positions, orientations, scales)


def test_contact_force_visualizer(monkeypatch):
    """Test arrow direction, scaling, thresholding, and tail offset."""
    monkeypatch.setattr(visualization, "VisualizationMarkers", _FakeVisualizationMarkers)
    cfg = BLUE_ARROW_X_MARKER_CFG.copy()
    cfg.markers["arrow"].scale = (0.04, 0.04, 0.2)
    visualizer = visualization.ContactForceVisualizer(cfg, force_scale=0.5)

    positions = torch.zeros((3, 3))
    forces = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.5, 0.0, 0.0]])
    visualizer.visualize(positions, forces, force_threshold=1.0)
    assert _FakeVisualizationMarkers.last_values is not None
    marker_positions, orientations, scales = _FakeVisualizationMarkers.last_values

    expected_directions = torch.nn.functional.normalize(forces[:2], dim=-1)
    local_x = torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1)
    torch.testing.assert_close(
        math_utils.quat_apply(orientations[:2], local_x),
        expected_directions,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        marker_positions,
        torch.tensor([[0.05, 0.0, 0.0], [0.0, 0.0, 0.075], [0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        scales,
        torch.tensor([[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [0.0, 0.0, 0.0]]),
    )
