# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared segmentation colorization (host vs Warp device parity)."""

import pytest

pytest.importorskip("warp")

import warp as wp

from isaaclab.renderers.segmentation_colors import (
    BACKGROUND_ID,
    UNLABELLED_ID,
    color_hash,
    pack_rgba,
    random_color_from_id,
    random_color_from_id_wp,
    unpack_rgba,
)


@wp.kernel
def _colorize_ids_kernel(ids: wp.array(dtype=wp.uint32), out: wp.array(dtype=wp.uint32)):
    i = wp.tid()
    out[i] = random_color_from_id_wp(ids[i])


def test_reserved_colors():
    """BACKGROUND is transparent black, UNLABELLED is opaque black."""
    assert random_color_from_id(BACKGROUND_ID) == (0, 0, 0, 0)
    assert random_color_from_id(UNLABELLED_ID) == (0, 0, 0, 255)
    assert pack_rgba(random_color_from_id(BACKGROUND_ID)) == 0
    assert pack_rgba(random_color_from_id(UNLABELLED_ID)) == 0xFF000000


def test_pack_rgba_is_little_endian():
    assert pack_rgba((1, 2, 3, 4)) == 1 | (2 << 8) | (3 << 16) | (4 << 24)


def test_unpack_rgba_is_inverse_of_pack():
    for color in [(0, 0, 0, 0), (255, 255, 255, 255), (1, 2, 3, 4), (0, 0, 0, 255)]:
        assert unpack_rgba(pack_rgba(color)) == color


def test_color_hash_is_uint32():
    for seed in (0, 1, 2, 12345, 0xFFFFFFFF):
        h = color_hash(seed)
        assert 0 <= h <= 0xFFFFFFFF


@pytest.mark.parametrize("device", ["cpu"])
def test_host_and_device_random_color_agree(device):
    """The host and Warp implementations must produce byte-identical colors for every id."""
    ids = list(range(256)) + [1000, 65535, 123456]
    ids_wp = wp.array(ids, dtype=wp.uint32, device=device)
    out_wp = wp.zeros(len(ids), dtype=wp.uint32, device=device)
    wp.launch(_colorize_ids_kernel, dim=len(ids), inputs=[ids_wp], outputs=[out_wp], device=device)

    device_colors = out_wp.numpy().tolist()
    host_colors = [pack_rgba(random_color_from_id(i)) for i in ids]
    assert device_colors == host_colors
