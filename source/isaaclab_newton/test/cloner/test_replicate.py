# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import warp as wp
from isaaclab_newton.cloner.replicate import _relative_clone_xform, _source_world_indices


def _default_quaternions(count: int) -> torch.Tensor:
    quaternions = torch.zeros((count, 4), dtype=torch.float32)
    quaternions[:, 3] = 1.0
    return quaternions


def _translation(transform: wp.transform) -> list[float]:
    return [float(value) for value in wp.transform_get_translation(transform)]


def _rotation(transform: wp.transform) -> list[float]:
    return [float(value) for value in wp.transform_get_rotation(transform)]


def test_source_world_indices_use_first_active_world():
    """Clone rows use their first active mapping column as the prototype source world."""
    mapping = torch.tensor(
        [
            [False, True, True],
            [False, False, False],
            [True, False, True],
        ],
        dtype=torch.bool,
    )

    assert _source_world_indices(mapping) == [1, -1, 0]


def test_relative_clone_xform_uses_source_world_pose():
    """A prototype parsed from world 0 and cloned into world 1 moves by world_1 * inverse(world_0)."""
    positions = torch.tensor([[2.0, 0.0, 0.0], [7.0, 0.0, 0.0]], dtype=torch.float32)
    quaternions = _default_quaternions(2)

    xform = _relative_clone_xform(0, 1, positions, quaternions)

    assert _translation(xform) == pytest.approx([5.0, 0.0, 0.0])
    assert _rotation(xform) == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_relative_clone_xform_identity_for_source_world():
    """Cloning a prototype into the world that already owns it produces identity."""
    positions = torch.tensor([[2.0, 0.0, 0.0], [7.0, 0.0, 0.0]], dtype=torch.float32)
    quaternions = _default_quaternions(2)

    xform = _relative_clone_xform(0, 0, positions, quaternions)

    assert _translation(xform) == pytest.approx([0.0, 0.0, 0.0])
    assert _rotation(xform) == pytest.approx([0.0, 0.0, 0.0, 1.0])
