# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for open-cable Warp reset-curve projection."""

from __future__ import annotations

import pytest
import torch
import warp as wp

from isaaclab_tasks.contrib.cable_routing.mdp.reset_curve_xpbd import (
    CableResetCurveXPBDCfg,
    relax_open_cable_curve_xpbd,
)

_BOARD_BOUNDS = ((-5.0, 5.0), (-5.0, 5.0))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"self_separation_distance": -0.1, "bend_radius": 0.1},
        {"self_separation_distance": 0.1, "bend_radius": float("nan")},
        {"self_separation_distance": 0.1, "bend_radius": 0.1, "iterations": 1.5},
        {"self_separation_distance": 0.1, "bend_radius": 0.1, "neighbor_exclusion": -1},
        {"self_separation_distance": 0.1, "bend_radius": 0.1, "taubin_smoothing_passes": 1.5},
        {"self_separation_distance": 0.1, "bend_radius": 0.1, "chebyshev_rho": 1.0},
        {"self_separation_distance": 0.1, "bend_radius": 0.1, "chebyshev_gamma": 0.0},
        {"self_separation_distance": 0.1, "bend_radius": 0.1, "cleanup_length_relaxation": 1.1},
    ),
)
def test_xpbd_cfg_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        CableResetCurveXPBDCfg(**kwargs)


def test_open_endpoints_are_nonlocal_self_separation_candidates() -> None:
    vertices = torch.tensor(
        (((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (0.0, 0.02)),),
        dtype=torch.float32,
    )
    cfg = CableResetCurveXPBDCfg(
        self_separation_distance=0.1,
        bend_radius=0.0,
        iterations=1,
        length_relaxation=0.0,
        chebyshev_acceleration=False,
        cleanup_iterations=0,
    )

    relaxed, _ = relax_open_cable_curve_xpbd(
        vertices,
        rest_length=1.0,
        board_bounds=_BOARD_BOUNDS,
        cfg=cfg,
    )

    endpoint_distance = torch.linalg.vector_norm(relaxed[0, 0] - relaxed[0, -1])
    torch.testing.assert_close(endpoint_distance, torch.tensor(0.1), atol=1.0e-6, rtol=0.0)
    # A closed-chain circular exclusion would incorrectly classify this pair as adjacent.
    assert not torch.equal(relaxed[0, 0], vertices[0, 0])
    assert not torch.equal(relaxed[0, -1], vertices[0, -1])


def test_fixed_waypoint_takes_precedence_and_fixed_geometry_is_projected() -> None:
    vertices = torch.tensor(
        (((-0.2, 0.0), (0.0, 0.0), (1.2, 0.4), (0.5, 0.5)),),
        dtype=torch.float32,
    )
    waypoint_indices = torch.tensor(((0,),))
    waypoint_positions = torch.tensor((((-0.2, 0.0),),), dtype=torch.float32)
    waypoint_mask = torch.tensor(((True,),))
    peg_centers = torch.zeros(1, 1, 2)
    cfg = CableResetCurveXPBDCfg(
        self_separation_distance=0.0,
        bend_radius=0.0,
        iterations=1,
        length_relaxation=0.0,
        chebyshev_acceleration=False,
        cleanup_iterations=0,
    )

    relaxed, diagnostics = relax_open_cable_curve_xpbd(
        vertices,
        rest_length=0.2,
        board_bounds=((-1.0, 1.0), (-1.0, 1.0)),
        cfg=cfg,
        waypoint_vertex_indices=waypoint_indices,
        waypoint_positions=waypoint_positions,
        waypoint_mask=waypoint_mask,
        peg_centers=peg_centers,
        peg_radii=0.2,
    )

    torch.testing.assert_close(relaxed[0, 0], waypoint_positions[0, 0], atol=0.0, rtol=0.0)
    assert torch.linalg.vector_norm(relaxed[0, 1]).item() >= 0.2 - 1.0e-6
    assert bool((relaxed[0, 2].abs() <= 1.0).all())
    torch.testing.assert_close(diagnostics.maximum_waypoint_error, torch.zeros(1), atol=0.0, rtol=0.0)
    torch.testing.assert_close(diagnostics.maximum_peg_penetration, torch.zeros(1), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(diagnostics.maximum_bounds_penetration, torch.zeros(1), atol=0.0, rtol=0.0)


def test_length_projection_and_cleanup_reduce_edge_error() -> None:
    vertices = torch.tensor(
        (((0.0, 0.0), (0.2, 0.0), (2.0, 0.0), (2.1, 0.2), (4.0, 0.0)),),
        dtype=torch.float32,
    )
    initial_error = (torch.linalg.vector_norm(vertices[:, 1:] - vertices[:, :-1], dim=-1) - 1.0).abs().amax()
    cfg = CableResetCurveXPBDCfg(
        self_separation_distance=0.0,
        bend_radius=0.0,
        iterations=50,
        chebyshev_acceleration=False,
        taubin_smoothing_passes=0,
        cleanup_iterations=10,
    )

    relaxed, diagnostics = relax_open_cable_curve_xpbd(
        vertices,
        rest_length=1.0,
        board_bounds=((-10.0, 10.0), (-10.0, 10.0)),
        cfg=cfg,
    )

    assert diagnostics.maximum_edge_length_error.item() < initial_error.item()
    assert diagnostics.maximum_edge_length_error.item() < 1.0e-4
    assert bool(torch.isfinite(relaxed).all())


def test_bend_regularization_reduces_discrete_turning() -> None:
    vertices = torch.tensor(
        (((-0.2, 0.0), (-0.1, 0.0), (0.0, 1.0), (0.1, 0.0), (0.2, 0.0)),),
        dtype=torch.float32,
    )

    def turning_energy(points: torch.Tensor) -> torch.Tensor:
        edge = torch.nn.functional.normalize(points[:, 1:] - points[:, :-1], dim=-1)
        return (1.0 - (edge[:, 1:] * edge[:, :-1]).sum(dim=-1)).sum(dim=-1)

    cfg = CableResetCurveXPBDCfg(
        self_separation_distance=0.0,
        bend_radius=1.0,
        iterations=10,
        length_relaxation=0.0,
        chebyshev_acceleration=False,
        cleanup_iterations=0,
    )
    relaxed, _ = relax_open_cable_curve_xpbd(
        vertices,
        rest_length=0.1,
        board_bounds=((-2.0, 2.0), (-2.0, 2.0)),
        cfg=cfg,
    )

    assert turning_energy(relaxed).item() < 0.1 * turning_energy(vertices).item()


def test_taubin_tail_reduces_curvature_noise_without_endpoint_or_waypoint_drift() -> None:
    vertices = torch.tensor(
        (
            (
                (-0.5, 0.0),
                (-0.4, 0.08),
                (-0.3, -0.08),
                (-0.2, 0.08),
                (-0.1, 0.0),
                (0.0, 0.0),
                (0.1, 0.0),
                (0.2, -0.08),
                (0.3, 0.08),
                (0.4, -0.08),
                (0.5, 0.0),
            ),
        ),
        dtype=torch.float32,
    )
    waypoint_indices = torch.tensor(((5,),))
    waypoint_positions = vertices[:, 5:6].clone()
    waypoint_mask = torch.ones((1, 1), dtype=torch.bool)
    common = {
        "rest_length": 0.1,
        "board_bounds": _BOARD_BOUNDS,
        "waypoint_vertex_indices": waypoint_indices,
        "waypoint_positions": waypoint_positions,
        "waypoint_mask": waypoint_mask,
    }

    unchanged, noisy_diagnostics = relax_open_cable_curve_xpbd(
        vertices,
        cfg=CableResetCurveXPBDCfg(
            self_separation_distance=0.0,
            bend_radius=0.0,
            iterations=0,
            taubin_smoothing_passes=0,
            cleanup_iterations=0,
        ),
        **common,
    )
    smoothed, smooth_diagnostics = relax_open_cable_curve_xpbd(
        vertices,
        cfg=CableResetCurveXPBDCfg(
            self_separation_distance=0.0,
            bend_radius=0.0,
            iterations=0,
            taubin_smoothing_passes=5,
            cleanup_iterations=0,
        ),
        **common,
    )

    torch.testing.assert_close(unchanged, vertices, atol=0.0, rtol=0.0)
    torch.testing.assert_close(smoothed[:, (0, -1)], vertices[:, (0, -1)], atol=0.0, rtol=0.0)
    torch.testing.assert_close(smoothed[:, 5], waypoint_positions[:, 0], atol=0.0, rtol=0.0)
    assert smooth_diagnostics.maximum_turning_angle.item() < noisy_diagnostics.maximum_turning_angle.item()


def test_taubin_tail_does_not_shrink_a_uniform_straight_open_chain() -> None:
    vertices = torch.stack(
        (torch.linspace(-0.5, 0.5, 11), torch.full((11,), 0.25)),
        dim=-1,
    ).unsqueeze(0)
    cfg = CableResetCurveXPBDCfg(
        self_separation_distance=0.0,
        bend_radius=0.0,
        iterations=0,
        taubin_smoothing_passes=5,
        cleanup_iterations=0,
    )

    smoothed, diagnostics = relax_open_cable_curve_xpbd(
        vertices,
        rest_length=0.1,
        board_bounds=_BOARD_BOUNDS,
        cfg=cfg,
    )

    torch.testing.assert_close(smoothed, vertices, atol=1.0e-7, rtol=0.0)
    torch.testing.assert_close(diagnostics.maximum_turning_angle, torch.zeros(1), atol=1.0e-7, rtol=0.0)


def test_accelerated_projection_is_deterministic_and_keeps_waypoints_exact() -> None:
    vertices = torch.tensor(
        (((0.0, 0.0), (0.1, 0.0), (0.2, 0.05), (0.3, 0.0), (0.4, 0.0)),),
        dtype=torch.float32,
    )
    waypoint_indices = torch.tensor(((0, 4),))
    waypoint_positions = vertices[:, (0, 4)].clone()
    waypoint_mask = torch.ones(1, 2, dtype=torch.bool)
    cfg = CableResetCurveXPBDCfg(
        self_separation_distance=0.0,
        bend_radius=0.2,
        iterations=30,
        cleanup_iterations=6,
    )
    kwargs = {
        "rest_length": 0.1,
        "board_bounds": ((-1.0, 1.0), (-1.0, 1.0)),
        "cfg": cfg,
        "waypoint_vertex_indices": waypoint_indices,
        "waypoint_positions": waypoint_positions,
        "waypoint_mask": waypoint_mask,
    }

    with torch.inference_mode():
        first, first_diagnostics = relax_open_cable_curve_xpbd(vertices, **kwargs)
    second, second_diagnostics = relax_open_cable_curve_xpbd(vertices, **kwargs)

    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first[:, (0, 4)], waypoint_positions, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        first_diagnostics.maximum_edge_length_error,
        second_diagnostics.maximum_edge_length_error,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Torch/Warp stream interop")
def test_cuda_projection_launches_on_current_torch_stream_without_device_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_stream = torch.cuda.Stream()
    launched_streams: list[int] = []
    original_launch = wp.launch

    def record_launch(*args, **kwargs):
        stream = kwargs.get("stream")
        assert stream is not None
        launched_streams.append(stream.cuda_stream)
        return original_launch(*args, **kwargs)

    def reject_device_sync(*args, **kwargs):
        raise AssertionError("Projection must not synchronize the complete CUDA device")

    monkeypatch.setattr(wp, "launch", record_launch)
    monkeypatch.setattr(wp, "synchronize_device", reject_device_sync)
    with torch.cuda.stream(torch_stream):
        expected_stream = torch.cuda.current_stream().cuda_stream
        vertices = torch.tensor(
            (((-0.2, 0.0), (0.0, 0.0), (0.35, 0.0)),),
            device="cuda",
            dtype=torch.float32,
        )
        relaxed, diagnostics = relax_open_cable_curve_xpbd(
            vertices,
            rest_length=0.1,
            board_bounds=((-0.3, 0.3), (-0.2, 0.2)),
            cfg=CableResetCurveXPBDCfg(
                self_separation_distance=0.01,
                bend_radius=0.02,
                iterations=2,
                cleanup_iterations=1,
            ),
            peg_centers=torch.zeros((1, 1, 2), device="cuda"),
            peg_radii=0.04,
        )
        finite = torch.isfinite(relaxed).all() & torch.isfinite(diagnostics.maximum_edge_length_error).all()

    torch_stream.synchronize()
    assert expected_stream != torch.cuda.default_stream().cuda_stream
    assert launched_streams
    assert set(launched_streams) == {expected_stream}
    assert bool(finite)


def test_input_contract_rejects_partial_waypoints_and_non_float32_vertices() -> None:
    cfg = CableResetCurveXPBDCfg(self_separation_distance=0.1, bend_radius=0.1)
    vertices = torch.zeros(1, 4, 2)
    with pytest.raises(ValueError, match="supplied together"):
        relax_open_cable_curve_xpbd(
            vertices,
            rest_length=0.1,
            board_bounds=_BOARD_BOUNDS,
            cfg=cfg,
            waypoint_vertex_indices=torch.zeros(1, 1, dtype=torch.long),
        )
    with pytest.raises(TypeError, match="float32"):
        relax_open_cable_curve_xpbd(
            vertices.double(),
            rest_length=0.1,
            board_bounds=_BOARD_BOUNDS,
            cfg=cfg,
        )
