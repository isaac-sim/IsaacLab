# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path

import pytest
import torch

from isaaclab.utils.math import quat_from_matrix

pytestmark = pytest.mark.unit


def _load_forge_utils_module():
    module_path = Path(__file__).parents[2] / "isaaclab_tasks" / "contrib" / "forge" / "forge_utils.py"
    spec = importlib.util.spec_from_file_location("forge_utils_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _random_rotation_matrices(num: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn((num, 3, 3), generator=generator, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    # Flip one column where needed so every matrix is a proper rotation.
    q[torch.linalg.det(q) < 0, :, 0] *= -1.0
    return q


def _wrench_in_frame(points, forces, rot, pos):
    """Reference wrench of point forces, expressed in the frame (rot, pos) and taken about its origin."""
    force = forces.sum(dim=1)
    torque = torch.cross(points - pos.unsqueeze(1), forces, dim=-1).sum(dim=1)
    rot_t = rot.transpose(-2, -1)
    return (rot_t @ force.unsqueeze(-1)).squeeze(-1), (rot_t @ torque.unsqueeze(-1)).squeeze(-1)


@pytest.mark.parametrize("rotate_frames", [False, True])
def test_change_FT_frame_matches_point_force_reference(rotate_frames):
    forge_utils = _load_forge_utils_module()
    generator = torch.Generator().manual_seed(0)
    num = 64

    # Two point forces per sample so the wrench has a non-trivial torque about any origin.
    points = torch.randn((num, 2, 3), generator=generator, dtype=torch.float64)
    forces = torch.randn((num, 2, 3), generator=generator, dtype=torch.float64)

    source_pos = torch.randn((num, 3), generator=generator, dtype=torch.float64)
    target_pos = torch.randn((num, 3), generator=generator, dtype=torch.float64)
    if rotate_frames:
        source_rot = _random_rotation_matrices(num, generator)
        target_rot = _random_rotation_matrices(num, generator)
    else:
        source_rot = torch.eye(3, dtype=torch.float64).expand(num, 3, 3)
        target_rot = source_rot

    source_F, source_T = _wrench_in_frame(points, forces, source_rot, source_pos)
    expected_F, expected_T = _wrench_in_frame(points, forces, target_rot, target_pos)

    target_F, target_T = forge_utils.change_FT_frame(
        source_F,
        source_T,
        (quat_from_matrix(source_rot), source_pos),
        (quat_from_matrix(target_rot), target_pos),
    )

    torch.testing.assert_close(target_F, expected_F)
    torch.testing.assert_close(target_T, expected_T)
