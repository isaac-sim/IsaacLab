# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The sample-set lens lookup textures, without a renderer."""

import math

import pytest
import torch
from isaaclab_fovi.sensors.lens_lut import build_lut_textures

pytestmark = pytest.mark.kitless


def test_exit_texture_marks_rays_outside_the_field_invalid():
    """The inverse lookup sends each in-field ray to its nearest sample, and every other ray to the invalid UV, never
    to the fovea."""
    # Three samples: the optical axis (OpenGL -z) and two rays 10 degrees off it.
    angle = math.radians(10.0)
    directions = torch.tensor(
        [[0.0, 0.0, -1.0], [math.sin(angle), 0.0, -math.cos(angle)], [0.0, math.sin(angle), -math.cos(angle)]]
    )
    uv = torch.tensor([[0.5, 0.5], [0.9, 0.5], [0.5, 0.9]])
    _, exit_texture = build_lut_textures(uv, directions, 8, 8, 32, 32, device="cpu")
    texels = exit_texture.reshape(-1, exit_texture.shape[-1])[:, :2]

    invalid = (texels == -1.0).all(dim=-1)
    assert invalid.any() and (~invalid).any()
    # Rays outside the 10 degree field outnumber those inside it on the hemisphere's octahedral grid.
    assert invalid.float().mean() > 0.5
    # Every valid texel holds one of the samples' UVs.
    assert all(any(torch.equal(texel, sample) for sample in uv) for texel in texels[~invalid])
    # The octahedral centre is the optical axis, which reaches the axis sample.
    centre = exit_texture[exit_texture.shape[0] // 2, exit_texture.shape[1] // 2, :2]
    torch.testing.assert_close(centre, uv[0])


def test_direction_texture_places_each_sample_by_its_uv():
    """A sample set with one sample per texel, in shuffled order, puts each sample's ray at its own texel."""
    size = 4
    axis = (torch.arange(size) + 0.5) / size
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    centres = torch.stack((col, row), -1).reshape(-1, 2)
    order = torch.randperm(size * size, generator=torch.Generator().manual_seed(0))
    directions = torch.nn.functional.normalize(torch.randn(size * size, 3, generator=torch.Generator().manual_seed(1)))
    direction_texture, _ = build_lut_textures(centres[order], directions, size, size, 16, 16, device="cpu")
    texels = (centres[order] * size).long()
    torch.testing.assert_close(direction_texture[texels[:, 1], texels[:, 0]], directions)
