# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for route-conditioned YAM reset-target geometry."""

from __future__ import annotations

import math

import pytest
import torch

from isaaclab.utils.math import matrix_from_quat

from isaaclab_tasks.contrib.cable_routing.mdp.reset_robot_targets import (
    CableResetRobotTargetCfg,
    build_top_down_yam_contact_target_poses,
    finite_reset_target_rows,
    select_downstream_cable_segment_indices,
    select_nearest_cable_segment_indices,
    valid_top_down_yam_target_rows,
)


def _straight_cable_poses(num_envs: int = 3, num_segments: int = 8) -> torch.Tensor:
    poses = torch.zeros(num_envs, num_segments, 7)
    poses[..., 0] = 0.1 * torch.arange(num_segments)
    poses[..., 6] = 1.0
    return poses


def test_reset_robot_target_cfg_has_runtime_defaults() -> None:
    cfg = CableResetRobotTargetCfg()

    assert cfg.enabled
    assert cfg.radial_cutoff == 0.05
    assert cfg.downstream_segment_offset == 1
    assert cfg.bimanual_segment_separation == 12
    assert cfg.reach_height == 0.055
    assert cfg.cage_height == 0.003
    assert cfg.cage_gripper_joint_position == 0.0045
    assert cfg.max_contact_position_error == 0.02
    assert cfg.min_tangent_alignment == 0.70
    assert cfg.post_settle_segment_window == 8
    assert cfg.ik_num_seeds == 4
    assert cfg.ik_noise_std == 0.35


def test_finite_reset_target_rows_rejects_poisoned_trials_without_raising() -> None:
    cable_poses = _straight_cable_poses(num_envs=4)
    contact_positions = torch.zeros(4, 3)
    contact_quaternions = torch.zeros(4, 4)
    contact_quaternions[:, 3] = 1.0
    robot_bases = torch.zeros(4, 2)
    cable_poses[1, 3, 0] = torch.nan
    contact_positions[2, 1] = torch.inf
    contact_quaternions[3, 0] = torch.nan
    robot_bases[3, 1] = torch.inf

    valid = finite_reset_target_rows(cable_poses, contact_positions, contact_quaternions, robot_bases)

    assert torch.equal(valid, torch.tensor((True, False, False, False)))


def test_valid_top_down_yam_target_rows_rejects_degenerate_cable_frames() -> None:
    root_half = math.sqrt(0.5)
    poses = torch.tensor(
        (
            (0.0, 0.0, 0.5, 0.0, root_half, 0.0, root_half),
            (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.5, torch.nan, 0.0, 0.0, 1.0),
        )
    )

    valid = valid_top_down_yam_target_rows(poses)

    assert torch.equal(valid, torch.tensor((True, False, False, False)))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"enabled": 1},
        {"radial_cutoff": 0.0},
        {"radial_cutoff": math.nan},
        {"downstream_segment_offset": 0},
        {"downstream_segment_offset": 1.5},
        {"bimanual_segment_separation": 0},
        {"reach_height": -0.01},
        {"reach_height": 0.001, "cage_height": 0.002},
        {"cage_gripper_joint_position": math.inf},
        {"max_contact_position_error": 0.0},
        {"min_tangent_alignment": 1.01},
        {"post_settle_segment_window": -1},
        {"post_settle_segment_window": 1.5},
        {"ik_num_seeds": 0},
        {"ik_num_seeds": 1.5},
        {"ik_noise_std": 0.0},
        {"ik_noise_std": math.nan},
    ),
)
def test_reset_robot_target_cfg_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        CableResetRobotTargetCfg(**kwargs)


def test_downstream_segment_selection_uses_last_local_segment_and_clamps() -> None:
    poses = _straight_cable_poses()
    pegs = torch.zeros(3, 2, 3)
    pegs[0, 0, 0] = 0.15
    pegs[1, 1, 0] = 0.65
    pegs[2, 0, 0] = 0.35

    selected = select_downstream_cable_segment_indices(
        poses,
        pegs,
        torch.tensor([0, 1, 0]),
        radial_cutoff=0.061,
        downstream_segment_offset=2,
    )

    # Last local segments are [2, 7, 4]. The middle result clamps safely at
    # the downstream endpoint instead of indexing past the cable.
    assert torch.equal(selected, torch.tensor([4, 7, 6]))


def test_top_down_targets_align_tangent_toward_each_robot_base() -> None:
    root_half = math.sqrt(0.5)
    selected_poses = torch.tensor(
        (
            (0.0, 0.0, 0.5, 0.0, root_half, 0.0, root_half),  # cable +Z maps to world +X
            (1.0, 2.0, 0.7, -root_half, 0.0, 0.0, root_half),  # cable +Z maps to world +Y
        )
    )
    robot_bases = torch.tensor(((-1.0, 0.0), (1.0, 3.0)))

    targets = build_top_down_yam_contact_target_poses(
        selected_poses,
        robot_bases,
        height_offsets=torch.tensor((0.055, 0.003)),
    )

    torch.testing.assert_close(
        targets[:, :3],
        torch.tensor(((0.0, 0.0, 0.555), (1.0, 2.0, 0.703))),
        atol=1.0e-6,
        rtol=0.0,
    )
    target_rotation = matrix_from_quat(targets[:, 3:7])
    expected_x = torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    expected_y = torch.tensor(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)))
    expected_z = torch.tensor(((0.0, 0.0, -1.0), (0.0, 0.0, -1.0)))
    torch.testing.assert_close(target_rotation[..., 0], expected_x, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(target_rotation[..., 1], expected_y, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(target_rotation[..., 2], expected_z, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(torch.linalg.vector_norm(targets[:, 3:7], dim=-1), torch.ones(2), atol=1.0e-6, rtol=0.0)
    assert bool(torch.isfinite(targets).all())


def test_top_down_targets_broadcast_scalar_height_offset() -> None:
    root_half = math.sqrt(0.5)
    selected_poses = torch.tensor(
        ((0.0, 0.0, 0.5, 0.0, root_half, 0.0, root_half),) * 2,
    )

    targets = build_top_down_yam_contact_target_poses(
        selected_poses,
        torch.tensor(((1.0, 0.0), (1.0, 0.0))),
        height_offsets=0.02,
    )

    torch.testing.assert_close(targets[:, 2], torch.full((2,), 0.52))


@pytest.mark.parametrize(
    ("poses", "pegs", "active_indices", "expected_error"),
    (
        (torch.zeros(2, 3, 6), torch.zeros(2, 1, 3), torch.zeros(2, dtype=torch.long), ValueError),
        (_straight_cable_poses(2), torch.zeros(1, 1, 3), torch.zeros(2, dtype=torch.long), ValueError),
        (_straight_cable_poses(2), torch.zeros(2, 1, 3), torch.zeros(2), TypeError),
        (_straight_cable_poses(2), torch.zeros(2, 1, 3), torch.tensor((0, 1)), ValueError),
    ),
)
def test_downstream_segment_selection_validates_shapes_and_indices(
    poses: torch.Tensor,
    pegs: torch.Tensor,
    active_indices: torch.Tensor,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        select_downstream_cable_segment_indices(
            poses,
            pegs,
            active_indices,
            radial_cutoff=0.05,
            downstream_segment_offset=1,
        )


def test_downstream_segment_selection_rejects_rows_without_local_cable() -> None:
    with pytest.raises(ValueError, match="rows \\[0, 1\\]"):
        select_downstream_cable_segment_indices(
            _straight_cable_poses(2),
            torch.full((2, 1, 3), 10.0),
            torch.zeros(2, dtype=torch.long),
            radial_cutoff=0.05,
            downstream_segment_offset=1,
        )


def test_nearest_segment_selection_tracks_local_material_sliding() -> None:
    positions = _straight_cable_poses(num_envs=2, num_segments=9)[..., :3]
    queries = torch.tensor(((0.62, 0.0, 0.0), (0.02, 0.0, 0.0)))

    selected = select_nearest_cable_segment_indices(
        positions,
        queries,
        torch.tensor((4, 4)),
        search_radius=2,
    )

    # The query at x=.62 is nearest segment 6. The second query would prefer
    # segment 0 globally, but the local material window correctly clamps it to 2.
    assert torch.equal(selected, torch.tensor((6, 2)))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"query_positions_w": torch.zeros(2, 2)},
        {"center_segment_indices": torch.zeros(2)},
        {"center_segment_indices": torch.tensor((0, 9))},
        {"search_radius": -1},
    ),
)
def test_nearest_segment_selection_validates_inputs(kwargs: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "cable_segment_positions_w": _straight_cable_poses(2, 9)[..., :3],
        "query_positions_w": torch.zeros(2, 3),
        "center_segment_indices": torch.tensor((0, 8)),
        "search_radius": 2,
    }
    arguments.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        select_nearest_cable_segment_indices(**arguments)


@pytest.mark.parametrize(
    ("poses", "robot_bases", "height_offsets", "message"),
    (
        (torch.zeros(2, 6), torch.zeros(2, 2), 0.0, "shape"),
        (torch.zeros(2, 7), torch.zeros(2, 3), 0.0, "shape"),
        (torch.zeros(2, 7), torch.zeros(2, 2), torch.zeros(2, 1), "height_offsets"),
        (torch.zeros(2, 7), torch.zeros(2, 2), 0.0, "non-zero norm"),
        (
            torch.tensor(((0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),) * 2),
            torch.zeros(2, 2),
            0.0,
            "planar component",
        ),
    ),
)
def test_top_down_target_builder_validates_inputs(
    poses: torch.Tensor,
    robot_bases: torch.Tensor,
    height_offsets: torch.Tensor | float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_top_down_yam_contact_target_poses(poses, robot_bases, height_offsets)
