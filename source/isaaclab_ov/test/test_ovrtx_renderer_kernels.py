# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX renderer kernels."""

import math

import numpy as np
import pytest
import warp as wp
from isaaclab_ov.renderers.ovrtx_renderer_kernels import (
    DEVICE,
    extract_all_depth_tiles_kernel,
    extract_all_depth_tiles_kernel_legacy,
    extract_all_rgba_tiles_kernel,
    generate_random_colors_from_ids_kernel,
    generate_random_colors_from_ids_kernel_legacy,
)


def _color_hash(seed: int) -> int:
    h = seed
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return h


def _random_colours_id(input_id: int) -> tuple[int, int, int, int]:
    GOLDEN_RATIO_INV = 1.0 / 1.618033988749895

    hash_val = _color_hash(input_id)
    hue = math.fmod(input_id * GOLDEN_RATIO_INV, 1.0)
    hue_perturbation = (hash_val & 0xFFFF) / 65536.0
    hue = math.fmod(hue + hue_perturbation * 0.1, 1.0)
    sat_hash = hash_val >> 16
    val_hash = hash_val >> 8
    saturation = 0.7 + 0.3 * ((sat_hash & 0xFF) / 255.0)
    value = 0.8 + 0.2 * ((val_hash & 0xFF) / 255.0)
    i = int(hue * 6.0)
    f = (hue * 6.0) - i
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - saturation * (1.0 - f))
    i = i % 6
    if i == 0:
        r, g, b = value, t, p
    elif i == 1:
        r, g, b = q, value, p
    elif i == 2:
        r, g, b = p, value, t
    elif i == 3:
        r, g, b = p, q, value
    elif i == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q
    return (int(r * 255), int(g * 255), int(b * 255), 255)


def _reference_color(input_id: int) -> int:
    if input_id == 0:
        return 0
    if input_id == 1:
        return 0xFF000000

    r, g, b, a = _random_colours_id(input_id)
    return r | (g << 8) | (b << 16) | (a << 24)


def _reference_extract_all_depth_tiles_legacy(
    tiled_2d: np.ndarray,
    num_envs: int,
    num_cols: int,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    """NumPy reference for ``extract_all_depth_tiles_kernel_legacy`` (2D tiled buffer)."""
    out = np.zeros((num_envs, tile_height, tile_width, 1), dtype=np.float32)
    for env_idx in range(num_envs):
        tile_x = env_idx % num_cols
        tile_y = env_idx // num_cols
        for y in range(tile_height):
            for x in range(tile_width):
                src_y = tile_y * tile_height + y
                src_x = tile_x * tile_width + x
                out[env_idx, y, x, 0] = tiled_2d[src_y, src_x]
    return out


def _reference_extract_all_depth_tiles(
    tiled_np: np.ndarray,
    num_envs: int,
    num_cols: int,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    """NumPy reference for ``extract_all_depth_tiles_kernel``."""
    return _reference_extract_all_depth_tiles_legacy(tiled_np[..., 0], num_envs, num_cols, tile_width, tile_height)


def _reference_extract_all_rgba_tiles(
    tiled_np: np.ndarray,
    num_envs: int,
    num_cols: int,
    tile_width: int,
    tile_height: int,
    num_channels: int,
) -> np.ndarray:
    """NumPy reference for ``extract_all_rgba_tiles_kernel``."""
    out_c = 4 if num_channels == 4 else 3
    out = np.zeros((num_envs, tile_height, tile_width, out_c), dtype=np.uint8)
    for env_idx in range(num_envs):
        tile_x = env_idx % num_cols
        tile_y = env_idx // num_cols
        for y in range(tile_height):
            for x in range(tile_width):
                src_y = tile_y * tile_height + y
                src_x = tile_x * tile_width + x
                out[env_idx, y, x, 0] = tiled_np[src_y, src_x, 0]
                out[env_idx, y, x, 1] = tiled_np[src_y, src_x, 1]
                out[env_idx, y, x, 2] = tiled_np[src_y, src_x, 2]
                if num_channels == 4:
                    out[env_idx, y, x, 3] = tiled_np[src_y, src_x, 3]
    return out


class TestExtractAllDepthTilesKernel:
    """Tests for ``extract_all_depth_tiles_kernel``."""

    def test_two_by_two_tile_grid(self):
        num_cols = 2
        num_envs = 4
        tile_width = 2
        tile_height = 3
        tiled_h = (num_envs // num_cols) * tile_height
        tiled_w = num_cols * tile_width
        tiled_np = np.zeros((tiled_h, tiled_w, 1), dtype=np.float32)
        for h in range(tiled_h):
            for w in range(tiled_w):
                tiled_np[h, w, 0] = float(h * 1000 + w)

        tiled_wp = wp.array(tiled_np, dtype=wp.float32, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 1), dtype=wp.float32, device=DEVICE)

        wp.launch(
            kernel=extract_all_depth_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_depth_tiles(tiled_np, num_envs, num_cols, tile_width, tile_height)
        np.testing.assert_allclose(output_wp.numpy(), expected, rtol=0, atol=0)

    def test_single_tile(self):
        num_cols = 1
        num_envs = 1
        tile_width = 4
        tile_height = 4
        tiled_np = np.arange(tile_height * tile_width, dtype=np.float32).reshape(tile_height, tile_width, 1)

        tiled_wp = wp.array(tiled_np, dtype=wp.float32, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 1), dtype=wp.float32, device=DEVICE)

        wp.launch(
            kernel=extract_all_depth_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_depth_tiles(tiled_np, num_envs, num_cols, tile_width, tile_height)
        np.testing.assert_array_equal(output_wp.numpy(), expected)

    @pytest.mark.parametrize(
        ("num_cols", "num_envs", "tile_width", "tile_height"),
        [
            (3, 6, 2, 2),
            (1, 3, 5, 1),
            (4, 8, 1, 1),
        ],
    )
    def test_various_layouts(self, num_cols, num_envs, tile_width, tile_height):
        num_rows = (num_envs + num_cols - 1) // num_cols
        tiled_h = num_rows * tile_height
        tiled_w = num_cols * tile_width
        rng = np.random.default_rng(12345)
        tiled_np = rng.random((tiled_h, tiled_w, 1), dtype=np.float32).astype(np.float32)

        tiled_wp = wp.array(tiled_np, dtype=wp.float32, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 1), dtype=wp.float32, device=DEVICE)

        wp.launch(
            kernel=extract_all_depth_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_depth_tiles(tiled_np, num_envs, num_cols, tile_width, tile_height)
        np.testing.assert_allclose(output_wp.numpy(), expected, rtol=1e-6, atol=1e-6)


class TestExtractAllDepthTilesKernelLegacy:
    """Tests for ``extract_all_depth_tiles_kernel_legacy`` (ovrtx < 0.3.0, 2D tiled buffer)."""

    def test_two_by_two_tile_grid(self):
        num_cols = 2
        num_envs = 4
        tile_width = 2
        tile_height = 3
        tiled_h = (num_envs // num_cols) * tile_height
        tiled_w = num_cols * tile_width
        tiled_np = np.zeros((tiled_h, tiled_w), dtype=np.float32)
        for h in range(tiled_h):
            for w in range(tiled_w):
                tiled_np[h, w] = float(h * 1000 + w)

        tiled_wp = wp.array(tiled_np, dtype=wp.float32, ndim=2, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 1), dtype=wp.float32, device=DEVICE)

        wp.launch(
            kernel=extract_all_depth_tiles_kernel_legacy,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_depth_tiles_legacy(tiled_np, num_envs, num_cols, tile_width, tile_height)
        np.testing.assert_allclose(output_wp.numpy(), expected, rtol=0, atol=0)

    def test_single_tile(self):
        num_cols = 1
        num_envs = 1
        tile_width = 4
        tile_height = 4
        tiled_np = np.arange(tile_height * tile_width, dtype=np.float32).reshape(tile_height, tile_width)

        tiled_wp = wp.array(tiled_np, dtype=wp.float32, ndim=2, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 1), dtype=wp.float32, device=DEVICE)

        wp.launch(
            kernel=extract_all_depth_tiles_kernel_legacy,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_depth_tiles_legacy(tiled_np, num_envs, num_cols, tile_width, tile_height)
        np.testing.assert_array_equal(output_wp.numpy(), expected)

    @pytest.mark.parametrize(
        ("num_cols", "num_envs", "tile_width", "tile_height"),
        [
            (3, 6, 2, 2),
            (1, 3, 5, 1),
            (4, 8, 1, 1),
        ],
    )
    def test_various_layouts(self, num_cols, num_envs, tile_width, tile_height):
        num_rows = (num_envs + num_cols - 1) // num_cols
        tiled_h = num_rows * tile_height
        tiled_w = num_cols * tile_width
        rng = np.random.default_rng(12345)
        tiled_np = rng.random((tiled_h, tiled_w), dtype=np.float32).astype(np.float32)

        tiled_wp = wp.array(tiled_np, dtype=wp.float32, ndim=2, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 1), dtype=wp.float32, device=DEVICE)

        wp.launch(
            kernel=extract_all_depth_tiles_kernel_legacy,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_depth_tiles_legacy(tiled_np, num_envs, num_cols, tile_width, tile_height)
        np.testing.assert_allclose(output_wp.numpy(), expected, rtol=1e-6, atol=1e-6)


class TestExtractAllRgbaTilesKernel:
    """Tests for ``extract_all_rgba_tiles_kernel``."""

    def test_two_by_two_tile_grid_rgba(self):
        num_cols = 2
        num_envs = 4
        tile_width = 2
        tile_height = 3
        num_channels = 4
        tiled_h = (num_envs // num_cols) * tile_height
        tiled_w = num_cols * tile_width
        tiled_np = np.zeros((tiled_h, tiled_w, 4), dtype=np.uint8)
        for h in range(tiled_h):
            for w in range(tiled_w):
                tiled_np[h, w, 0] = (h * 17 + w) % 256
                tiled_np[h, w, 1] = (h * 31 + w * 3) % 256
                tiled_np[h, w, 2] = (h + w * 11) % 256
                tiled_np[h, w, 3] = (h * 7 + w * 13) % 256

        tiled_wp = wp.array(tiled_np, dtype=wp.uint8, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, num_channels), dtype=wp.uint8, device=DEVICE)

        wp.launch(
            kernel=extract_all_rgba_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height, num_channels],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_rgba_tiles(
            tiled_np, num_envs, num_cols, tile_width, tile_height, num_channels
        )
        np.testing.assert_array_equal(output_wp.numpy(), expected)

    def test_single_tile_rgb(self):
        num_cols = 1
        num_envs = 1
        tile_width = 4
        tile_height = 4
        num_channels = 3
        tiled_np = np.arange(tile_height * tile_width * 3, dtype=np.uint8).reshape(tile_height, tile_width, 3)

        tiled_wp = wp.array(tiled_np, dtype=wp.uint8, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, num_channels), dtype=wp.uint8, device=DEVICE)

        wp.launch(
            kernel=extract_all_rgba_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height, num_channels],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_rgba_tiles(
            tiled_np, num_envs, num_cols, tile_width, tile_height, num_channels
        )
        np.testing.assert_array_equal(output_wp.numpy(), expected)

    def test_num_channels_not_four_skips_alpha(self):
        """Values other than 4 use the RGB-only path (same as RGB tiled input)."""
        num_cols = 1
        num_envs = 1
        tile_width = 2
        tile_height = 2
        tiled_np = np.array(
            [
                [[1, 2, 3, 99], [4, 5, 6, 88]],
                [[7, 8, 9, 77], [10, 11, 12, 66]],
            ],
            dtype=np.uint8,
        )

        tiled_wp = wp.array(tiled_np, dtype=wp.uint8, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(1, 2, 2, 3), dtype=wp.uint8, device=DEVICE)

        wp.launch(
            kernel=extract_all_rgba_tiles_kernel,
            dim=(1, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height, 2],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_rgba_tiles(tiled_np, num_envs, num_cols, tile_width, tile_height, 2)
        np.testing.assert_array_equal(output_wp.numpy(), expected)

    @pytest.mark.parametrize(
        ("num_cols", "num_envs", "tile_width", "tile_height", "num_channels"),
        [
            (3, 6, 2, 2, 3),
            (3, 6, 2, 2, 4),
            (1, 3, 5, 1, 3),
            (4, 8, 1, 1, 4),
        ],
    )
    def test_various_layouts(self, num_cols, num_envs, tile_width, tile_height, num_channels):
        num_rows = (num_envs + num_cols - 1) // num_cols
        tiled_h = num_rows * tile_height
        tiled_w = num_cols * tile_width
        c_in = 4 if num_channels == 4 else 3
        rng = np.random.default_rng(24680)
        tiled_np = rng.integers(0, 256, size=(tiled_h, tiled_w, c_in), dtype=np.uint8)

        tiled_wp = wp.array(tiled_np, dtype=wp.uint8, ndim=3, device=DEVICE)
        output_wp = wp.zeros(
            shape=(num_envs, tile_height, tile_width, num_channels),
            dtype=wp.uint8,
            device=DEVICE,
        )

        wp.launch(
            kernel=extract_all_rgba_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height, num_channels],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_rgba_tiles(
            tiled_np, num_envs, num_cols, tile_width, tile_height, num_channels
        )
        np.testing.assert_array_equal(output_wp.numpy(), expected)


class TestRandomColorsFromIdsKernel:
    """Tests for generate_random_colors_from_ids_kernel."""

    def test_random_colors(self):
        inputs_np = np.array([[[0], [1]], [[2], [3]]], dtype=np.uint32)
        input_ids = wp.array(inputs_np, dtype=wp.uint32, ndim=3, device=DEVICE)
        output_colors = wp.zeros(shape=inputs_np.shape, dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel,
            dim=inputs_np.shape,
            inputs=[input_ids, output_colors],
            device=DEVICE,
        )
        wp.synchronize()

        out_np = output_colors.numpy()
        for (i, j, k), input_id in np.ndenumerate(inputs_np):
            input_id = int(np.uint32(input_id))
            ref_color = _reference_color(input_id)
            out_color = int(out_np[i, j, k])
            assert out_color == ref_color, (
                f"At ({i},{j},{k}) id={input_id}: expected 0x{ref_color:08x}, got 0x{out_color:08x}"
            )

    def test_deterministic_across_launches(self):
        shape = (4, 4, 1)
        rng = np.random.default_rng(42)
        inputs_np = rng.integers(0, 2**31, size=shape, dtype=np.uint32)
        input_ids = wp.array(inputs_np, dtype=wp.uint32, ndim=3, device=DEVICE)
        output_colors = wp.zeros(shape=shape, dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel,
            dim=shape,
            inputs=[input_ids, output_colors],
            device=DEVICE,
        )
        wp.synchronize()
        first_run = output_colors.numpy().copy()

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel,
            dim=shape,
            inputs=[input_ids, output_colors],
            device=DEVICE,
        )
        wp.synchronize()
        second_run = output_colors.numpy()

        np.testing.assert_array_equal(first_run, second_run)

    @pytest.mark.parametrize(
        "input_value",
        [
            0,
            1,
            2,
            3,
            100,
        ],
    )
    def test_single_value(self, input_value):
        inputs_np = np.array([[[input_value]]], dtype=np.uint32)
        input_ids = wp.array(inputs_np, dtype=wp.uint32, ndim=3, device=DEVICE)
        output_colors = wp.zeros(shape=(1, 1, 1), dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel,
            dim=(1, 1, 1),
            inputs=[input_ids, output_colors],
            device=DEVICE,
        )
        wp.synchronize()

        ref_color = _reference_color(int(np.uint32(input_value)))
        out_color = int(output_colors.numpy()[0, 0, 0])
        assert out_color == ref_color, (
            f"id=0x{int(np.uint32(input_value)):08x}: expected 0x{ref_color:08x}, got 0x{out_color:08x}"
        )


class TestRandomColorsFromIdsKernelLegacy:
    """Tests for ``generate_random_colors_from_ids_kernel_legacy`` (ovrtx < 0.3.0, 2D buffers)."""

    def test_random_colors(self):
        inputs_np = np.array([[0, 1], [2, 3]], dtype=np.uint32)
        input_ids = wp.array(inputs_np, dtype=wp.uint32, ndim=2, device=DEVICE)
        output_colors = wp.zeros(shape=inputs_np.shape, dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel_legacy,
            dim=inputs_np.shape,
            inputs=[input_ids, output_colors],
            device=DEVICE,
        )
        wp.synchronize()

        out_np = output_colors.numpy()
        for (i, j), input_id in np.ndenumerate(inputs_np):
            input_id = int(np.uint32(input_id))
            ref_color = _reference_color(input_id)
            out_color = int(out_np[i, j])
            assert out_color == ref_color, (
                f"At ({i},{j}) id={input_id}: expected 0x{ref_color:08x}, got 0x{out_color:08x}"
            )

    def test_deterministic_across_launches(self):
        shape = (4, 4)
        rng = np.random.default_rng(42)
        inputs_np = rng.integers(0, 2**31, size=shape, dtype=np.uint32)
        input_ids = wp.array(inputs_np, dtype=wp.uint32, ndim=2, device=DEVICE)
        output_colors = wp.zeros(shape=shape, dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel_legacy,
            dim=shape,
            inputs=[input_ids, output_colors],
            device=DEVICE,
        )
        wp.synchronize()
        first_run = output_colors.numpy().copy()

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel_legacy,
            dim=shape,
            inputs=[input_ids, output_colors],
            device=DEVICE,
        )
        wp.synchronize()
        second_run = output_colors.numpy()

        np.testing.assert_array_equal(first_run, second_run)

    @pytest.mark.parametrize(
        "input_value",
        [
            0,
            1,
            2,
            3,
            100,
        ],
    )
    def test_single_value(self, input_value):
        inputs_np = np.array([[input_value]], dtype=np.uint32)
        input_ids = wp.array(inputs_np, dtype=wp.uint32, ndim=2, device=DEVICE)
        output_colors = wp.zeros(shape=(1, 1), dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel_legacy,
            dim=(1, 1),
            inputs=[input_ids, output_colors],
            device=DEVICE,
        )
        wp.synchronize()

        ref_color = _reference_color(int(np.uint32(input_value)))
        out_color = int(output_colors.numpy()[0, 0])
        assert out_color == ref_color, (
            f"id=0x{int(np.uint32(input_value)):08x}: expected 0x{ref_color:08x}, got 0x{out_color:08x}"
        )
