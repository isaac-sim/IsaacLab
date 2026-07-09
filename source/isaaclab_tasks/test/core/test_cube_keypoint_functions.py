# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for the Warp-based cube keypoint function.

The Warp implementation replaced a validated torch implementation; every test
compares against a torch reference copied verbatim from the replaced code.
"""

import pytest
import torch

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import compute_cube_keypoints

_DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def _reference_compute_cube_keypoints(
    pose: torch.Tensor,
    num_keypoints: int = 8,
    size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03),
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    num_envs = pose.shape[0]
    if out is None:
        out = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    else:
        out[:] = 1.0
    for i in range(num_keypoints):
        positive_axes = [((i >> axis) & 1) == 0 for axis in range(3)]
        corner_values = ([(1 if positive_axes[axis] else -1) * side / 2 for axis, side in enumerate(size)],)
        corner = torch.tensor(corner_values, dtype=torch.float32, device=pose.device) * out[:, i, :]
        out[:, i, :] = pose[:, :3] + quat_apply(pose[:, 3:7], corner)
    return out


def _random_poses(generator: torch.Generator, num: int, device: str) -> torch.Tensor:
    positions = torch.randn((num, 3), generator=generator, device=device)
    quats = torch.randn((num, 4), generator=generator, device=device)
    quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)
    return torch.cat((positions, quats), dim=-1)


@pytest.mark.parametrize("device", _DEVICES)
def test_keypoints_match_torch_reference(device):
    generator = torch.Generator(device=device).manual_seed(11)
    pose = _random_poses(generator, 256, device)

    expected = _reference_compute_cube_keypoints(pose)
    actual = compute_cube_keypoints(pose)

    assert actual.shape == (256, 8, 3)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_keypoints_match_reference_for_custom_size(device):
    generator = torch.Generator(device=device).manual_seed(12)
    pose = _random_poses(generator, 64, device)
    size = (0.1, 0.04, 0.02)

    expected = _reference_compute_cube_keypoints(pose, size=size)
    actual = compute_cube_keypoints(pose, size=size)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_keypoints_write_into_provided_buffer(device):
    generator = torch.Generator(device=device).manual_seed(13)
    pose = _random_poses(generator, 64, device)
    out = torch.full((64, 8, 3), 7.0, dtype=torch.float32, device=device)

    result = compute_cube_keypoints(pose, out=out)

    assert result is out
    expected = _reference_compute_cube_keypoints(pose)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_keypoints_identity_pose_gives_half_side_corners(device):
    pose = torch.zeros((1, 7), device=device)
    pose[:, 6] = 1.0  # identity quaternion (x, y, z, w)

    corners = compute_cube_keypoints(pose)

    assert torch.allclose(corners.abs(), torch.full((1, 8, 3), 0.03, device=device), atol=1e-6)
    # all eight sign combinations must be present exactly once
    signs = {tuple(int(v) for v in torch.sign(corner).tolist()) for corner in corners[0]}
    assert len(signs) == 8


@pytest.mark.parametrize("device", _DEVICES)
def test_deprecated_shim_delegates_and_warns(device):
    from isaaclab_tasks.core.reorient.config.shadow_hand import shadow_hand_camera_env

    generator = torch.Generator(device=device).manual_seed(14)
    pose = _random_poses(generator, 16, device)

    with pytest.warns(DeprecationWarning):
        shimmed = shadow_hand_camera_env.compute_keypoints(pose)

    torch.testing.assert_close(shimmed, compute_cube_keypoints(pose), atol=1e-6, rtol=1e-6)
