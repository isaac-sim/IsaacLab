# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for Franka Pour geometric termination predicates."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.contrib.franka_pour.mdp import terminations


def _identity_pose(position: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor((*position, 0.0, 0.0, 0.0, 1.0), dtype=torch.float32).unsqueeze(0)


def _source_receiver_overlap(source_pose: torch.Tensor) -> torch.Tensor:
    """Evaluate the public overlap term with the task's cup geometry."""
    cfg = SimpleNamespace(
        source_cup_inner_width=0.042,
        source_cup_inner_depth=0.042,
        source_cup_cavity_depth=0.110,
        source_cup_wall_thickness=0.007,
        source_cup_bottom_thickness=0.009,
        target_cup_inner_width=0.140,
        target_cup_inner_depth=0.140,
        target_cup_cavity_depth=0.065,
        target_cup_wall_thickness=0.009,
        target_cup_bottom_thickness=0.009,
    )
    target_pose = _identity_pose((0.0, 0.0, 0.0)).expand(source_pose.shape[0], -1)
    env = SimpleNamespace(
        cfg=cfg,
        cup_pose_e=lambda: source_pose,
        target_pose_e=lambda: target_pose,
    )
    return terminations.source_receiver_overlap(env, clearance=0.003)


def test_source_receiver_envelope_overlap_rejects_contact_but_allows_safe_pour_clearance():
    source_pose = torch.cat(
        (
            _identity_pose((0.0, 0.0, 0.070)),
            _identity_pose((0.0, 0.0, 0.080)),
            _identity_pose((0.20, 0.0, 0.0)),
        )
    )

    assert torch.equal(_source_receiver_overlap(source_pose), torch.tensor((True, False, False)))


def test_source_receiver_envelope_overlap_accounts_for_source_tilt():
    half_sqrt = 2.0**-0.5
    source_pose = torch.tensor(
        (
            (0.0, 0.0, 0.080, half_sqrt, 0.0, 0.0, half_sqrt),
            (0.0, 0.0, 0.150, half_sqrt, 0.0, 0.0, half_sqrt),
        ),
        dtype=torch.float32,
    )

    assert torch.equal(_source_receiver_overlap(source_pose), torch.tensor((True, False)))
