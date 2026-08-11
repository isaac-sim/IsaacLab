# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for workspace-aware cable contact selection during reset authoring."""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.contrib.cable_routing.mdp.reset_robot_targets import (
    select_downstream_cable_segment_indices,
    select_workspace_aware_cable_contact_indices,
)


def _cable_poses(segment_xy: list[tuple[float, float]], num_rows: int = 1) -> torch.Tensor:
    """Create identity-oriented cable poses from planar segment centers."""
    poses = torch.zeros(num_rows, len(segment_xy), 7)
    poses[..., :2] = torch.tensor(segment_xy)[None]
    poses[..., 6] = 1.0
    return poses


def test_workspace_selection_prefers_reachable_free_strand_over_fixed_offset() -> None:
    """The current peg-0 geometry should not force the unreachable fixed +6 contact."""
    segment_xy = [
        (0.045, -0.055),
        (0.055, -0.035),
        (0.075, -0.055),
        (0.120, 0.120),
        (-0.020, 0.150),
        (-0.120, 0.160),
        (-0.100, -0.160),
        (-0.180, -0.200),
        (0.150, -0.050),
        (-0.050, 0.000),
        (-0.140, -0.160),
        (0.100, 0.190),
    ]
    cable_poses = _cable_poses(segment_xy)
    peg_positions = torch.tensor([[[0.045, -0.055, 0.0], [-0.035, 0.085, 0.0]]])
    robot_bases = torch.tensor([[[-0.335, 0.300], [-0.335, -0.300]]])

    fixed_segment = select_downstream_cable_segment_indices(
        cable_poses,
        peg_positions,
        torch.tensor([0]),
        radial_cutoff=0.05,
        downstream_segment_offset=6,
    )
    selected_segment, selected_arm = select_workspace_aware_cable_contact_indices(
        cable_poses,
        peg_positions,
        torch.tensor([0]),
        robot_bases,
        radial_cutoff=0.05,
        maximum_planar_reach=0.35,
    )

    assert fixed_segment.item() == 8
    assert selected_segment.item() == 7
    assert selected_arm.item() == 1
    fixed_distance = torch.linalg.vector_norm(robot_bases[0] - cable_poses[0, fixed_segment, :2], dim=-1).amin()
    selected_distance = torch.linalg.vector_norm(robot_bases[0, selected_arm] - cable_poses[0, selected_segment, :2])
    assert fixed_distance > 0.35
    assert selected_distance < 0.20


def test_workspace_selection_avoids_other_pegs_and_falls_back_when_necessary() -> None:
    cable_poses = _cable_poses(
        [(0.0, 0.0), (0.01, 0.0), (-0.80, 0.0), (-0.90, 0.0)],
        num_rows=2,
    )
    peg_positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [-0.90, 0.0, 0.0], [3.0, 3.0, 0.0]],
            [[0.0, 0.0, 0.0], [-0.80, 0.0, 0.0], [-0.90, 0.0, 0.0]],
        ]
    )
    robot_bases = torch.tensor(
        [
            [[-1.0, 0.0], [5.0, 5.0]],
            [[-1.0, 0.0], [5.0, 5.0]],
        ]
    )

    segment, arm = select_workspace_aware_cable_contact_indices(
        cable_poses,
        peg_positions,
        torch.tensor([0, 0]),
        robot_bases,
        radial_cutoff=0.05,
    )

    # Row 0 excludes the closest segment because it is local to peg 1. Row 1
    # has no peg-free downstream strand, so it safely falls back to the closest
    # downstream contact and leaves strict IK to accept or reject it.
    assert torch.equal(segment, torch.tensor([2, 3]))
    assert torch.equal(arm, torch.tensor([0, 0]))


def test_workspace_selection_has_deterministic_lexicographic_ties() -> None:
    cable_poses = _cable_poses([(0.0, 0.0), (1.0, 0.0), (1.0, 0.0)])
    peg_positions = torch.tensor([[[0.0, 0.0, 0.0]]])
    robot_bases = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]])
    arguments = (
        cable_poses,
        peg_positions,
        torch.tensor([0]),
        robot_bases,
    )

    first = select_workspace_aware_cable_contact_indices(*arguments, radial_cutoff=0.05)
    second = select_workspace_aware_cable_contact_indices(*arguments, radial_cutoff=0.05)

    assert torch.equal(first[0], torch.tensor([1]))
    assert torch.equal(first[1], torch.tensor([0]))
    assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(first, second))


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"active_peg_indices": torch.tensor([0.0])}, "integer dtype"),
        ({"robot_base_xy_w": torch.zeros(1, 2, 3)}, "shape"),
        ({"minimum_downstream_offset": 0}, "positive integer"),
        ({"maximum_planar_reach": 0.0}, "finite and positive"),
        ({"maximum_planar_reach": 0.1}, "within maximum_planar_reach"),
    ),
)
def test_workspace_selection_validates_inputs(override: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "cable_segment_poses_w": _cable_poses([(0.0, 0.0), (1.0, 0.0)]),
        "peg_positions_w": torch.tensor([[[0.0, 0.0, 0.0]]]),
        "active_peg_indices": torch.tensor([0]),
        "robot_base_xy_w": torch.tensor([[[-1.0, 0.0], [-1.0, 1.0]]]),
        "radial_cutoff": 0.05,
    }
    arguments.update(override)

    with pytest.raises((TypeError, ValueError), match=message):
        select_workspace_aware_cable_contact_indices(**arguments)


def test_workspace_selection_rejects_missing_local_or_downstream_segments() -> None:
    with pytest.raises(ValueError, match="active peg for rows \\[0\\]"):
        select_workspace_aware_cable_contact_indices(
            _cable_poses([(0.0, 0.0), (1.0, 0.0)]),
            torch.tensor([[[10.0, 0.0, 0.0]]]),
            torch.tensor([0]),
            torch.tensor([[[-1.0, 0.0]]]),
            radial_cutoff=0.05,
        )

    with pytest.raises(ValueError, match="downstream offset for rows \\[0\\]"):
        select_workspace_aware_cable_contact_indices(
            _cable_poses([(1.0, 0.0), (0.0, 0.0)]),
            torch.tensor([[[0.0, 0.0, 0.0]]]),
            torch.tensor([0]),
            torch.tensor([[[-1.0, 0.0]]]),
            radial_cutoff=0.05,
        )
