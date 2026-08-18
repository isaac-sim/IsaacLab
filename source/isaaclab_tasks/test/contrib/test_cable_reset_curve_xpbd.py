# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the cable reset-curve projector."""

import torch

from isaaclab_tasks.contrib.cable_routing.mdp.reset_curve_xpbd import (
    CableResetCurveXPBDCfg,
    relax_open_cable_curve_xpbd,
)


def test_projection_preserves_waypoints_and_enforces_geometry() -> None:
    vertices = torch.tensor(
        (((-0.2, 0.0), (0.0, 0.0), (1.2, 0.4), (0.5, 0.5)),),
        dtype=torch.float32,
    )
    waypoint_indices = torch.tensor(((0,),))
    waypoint_positions = vertices[:, :1].clone()
    relaxed = relax_open_cable_curve_xpbd(
        vertices,
        rest_length=0.2,
        board_bounds=((-1.0, 1.0), (-1.0, 1.0)),
        cfg=CableResetCurveXPBDCfg(
            self_separation_distance=0.0,
            bend_radius=0.0,
            iterations=1,
            length_relaxation=0.0,
            chebyshev_acceleration=False,
            cleanup_iterations=0,
        ),
        waypoint_vertex_indices=waypoint_indices,
        waypoint_positions=waypoint_positions,
        waypoint_mask=torch.ones(1, 1, dtype=torch.bool),
        peg_centers=torch.zeros(1, 1, 2),
        peg_radii=0.2,
    )

    torch.testing.assert_close(relaxed[:, :1], waypoint_positions, atol=0.0, rtol=0.0)
    assert torch.linalg.vector_norm(relaxed[0, 1]).item() >= 0.2 - 1.0e-6
    assert bool((relaxed.abs() <= 1.0).all())


def test_projection_reduces_edge_error_and_is_deterministic() -> None:
    vertices = torch.tensor(
        (((0.0, 0.0), (0.2, 0.0), (2.0, 0.0), (2.1, 0.2), (4.0, 0.0)),),
        dtype=torch.float32,
    )
    initial_error = (torch.linalg.vector_norm(vertices[:, 1:] - vertices[:, :-1], dim=-1) - 1.0).abs().amax()
    kwargs = {
        "rest_length": 1.0,
        "board_bounds": ((-10.0, 10.0), (-10.0, 10.0)),
        "cfg": CableResetCurveXPBDCfg(
            self_separation_distance=0.0,
            bend_radius=0.0,
            iterations=50,
            chebyshev_acceleration=False,
            taubin_smoothing_passes=0,
            cleanup_iterations=10,
        ),
    }

    first = relax_open_cable_curve_xpbd(vertices, **kwargs)
    second = relax_open_cable_curve_xpbd(vertices, **kwargs)
    final_error = (torch.linalg.vector_norm(first[:, 1:] - first[:, :-1], dim=-1) - 1.0).abs().amax()

    assert final_error < initial_error
    assert final_error < 1.0e-4
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
