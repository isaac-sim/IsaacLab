# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for route-conditioned manipulator reset targets."""

import math

import torch

from isaaclab.utils.math import matrix_from_quat

from isaaclab_tasks.contrib.cable_routing.mdp.reset_robot_targets import (
    build_top_down_contact_target_poses,
    select_nearest_cable_segment_indices,
    select_workspace_aware_cable_contact_indices,
)


def _straight_cable_poses(num_envs: int = 2, num_segments: int = 8) -> torch.Tensor:
    poses = torch.zeros(num_envs, num_segments, 7)
    poses[..., 0] = 0.1 * torch.arange(num_segments)
    poses[..., 6] = 1.0
    return poses


def test_workspace_selection_chooses_reachable_free_strand_and_arm() -> None:
    cable = _straight_cable_poses(num_envs=1)
    pegs = torch.tensor([[[0.1, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    robot_bases = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]])

    segment, arm = select_workspace_aware_cable_contact_indices(
        cable,
        pegs,
        torch.tensor([0]),
        robot_bases,
        radial_cutoff=0.06,
    )

    assert torch.equal(segment, torch.tensor([7]))
    assert torch.equal(arm, torch.tensor([1]))


def test_nearest_segment_selection_tracks_local_material_sliding() -> None:
    positions = _straight_cable_poses(num_envs=2, num_segments=9)[..., :3]
    selected = select_nearest_cable_segment_indices(
        positions,
        torch.tensor(((0.62, 0.0, 0.0), (0.02, 0.0, 0.0))),
        torch.tensor((4, 4)),
        search_radius=2,
    )

    assert torch.equal(selected, torch.tensor((6, 2)))


def test_top_down_targets_align_tangent_toward_robot_base() -> None:
    root_half = math.sqrt(0.5)
    cable = torch.tensor(
        (
            (0.0, 0.0, 0.5, 0.0, root_half, 0.0, root_half),
            (1.0, 2.0, 0.7, -root_half, 0.0, 0.0, root_half),
        )
    )
    targets = build_top_down_contact_target_poses(
        cable,
        torch.tensor(((-1.0, 0.0), (1.0, 3.0))),
        height_offsets=torch.tensor((0.055, 0.003)),
    )

    torch.testing.assert_close(
        targets[:, :3],
        torch.tensor(((0.0, 0.0, 0.555), (1.0, 2.0, 0.703))),
        atol=1.0e-6,
        rtol=0.0,
    )
    rotation = matrix_from_quat(targets[:, 3:7])
    torch.testing.assert_close(
        rotation[..., 0],
        torch.tensor(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rotation[..., 2],
        torch.tensor(((0.0, 0.0, -1.0), (0.0, 0.0, -1.0))),
        atol=1.0e-6,
        rtol=0.0,
    )
