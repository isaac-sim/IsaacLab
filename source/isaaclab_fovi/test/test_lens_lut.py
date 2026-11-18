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
