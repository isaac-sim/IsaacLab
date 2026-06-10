# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import warp as wp
from isaaclab_newton.cloner.replicate import _relative_clone_xform


def _default_quaternions(count: int) -> torch.Tensor:
    quaternions = torch.zeros((count, 4), dtype=torch.float32)
    quaternions[:, 3] = 1.0
    return quaternions


def _translation(transform: wp.transform) -> list[float]:
    return [float(value) for value in wp.transform_get_translation(transform)]


def _rotation(transform: wp.transform) -> list[float]:
    return [float(value) for value in wp.transform_get_rotation(transform)]


def test_relative_clone_xform_uses_source_env_pose():
    """A prototype parsed from env_2 and cloned into env_7 moves by env_7 * inverse(env_2)."""
    positions = torch.tensor([[2.0, 0.0, 0.0], [7.0, 0.0, 0.0]], dtype=torch.float32)
    quaternions = _default_quaternions(2)
    env_id_to_world_index = {2: 0, 7: 1}

    xform = _relative_clone_xform(
        "/World/envs/env_2/Object",
        "/World/envs/env_{}/Object",
        1,
        env_id_to_world_index,
        positions,
        quaternions,
    )

    assert _translation(xform) == pytest.approx([5.0, 0.0, 0.0])
    assert _rotation(xform) == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_relative_clone_xform_identity_for_source_world():
    """Cloning a prototype into the world that already owns it produces identity."""
    positions = torch.tensor([[2.0, 0.0, 0.0], [7.0, 0.0, 0.0]], dtype=torch.float32)
    quaternions = _default_quaternions(2)
    env_id_to_world_index = {2: 0, 7: 1}

    xform = _relative_clone_xform(
        "/World/envs/env_2/Object",
        "/World/envs/env_{}/Object",
        0,
        env_id_to_world_index,
        positions,
        quaternions,
    )

    assert _translation(xform) == pytest.approx([0.0, 0.0, 0.0])
    assert _rotation(xform) == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_relative_clone_xform_requires_source_env_in_mapping():
    """The source env id recovered from the path must be present in the queued env ids."""
    positions = torch.tensor([[2.0, 0.0, 0.0], [7.0, 0.0, 0.0]], dtype=torch.float32)
    quaternions = _default_quaternions(2)
    env_id_to_world_index = {2: 0, 7: 1}

    with pytest.raises(RuntimeError, match="not present in env_ids"):
        _relative_clone_xform(
            "/World/envs/env_9/Object",
            "/World/envs/env_{}/Object",
            1,
            env_id_to_world_index,
            positions,
            quaternions,
        )
