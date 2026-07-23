# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX renderer kernels."""

import numpy as np
import pytest
import warp as wp
from isaaclab_ov.renderers.ovrtx_renderer_kernels import (
    extract_all_tiles_kernel,
    generate_random_colors_from_ids_kernel,
)

from isaaclab.renderers.segmentation_colors import pack_rgba, random_color_from_id

DEVICE = "cuda:0"


def _reference_extract_all_depth_tiles(
    tiled_np: np.ndarray,
    num_envs: int,
    num_cols: int,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    """NumPy reference for ``extract_all_depth_tiles_kernel``."""
    out = np.zeros((num_envs, tile_height, tile_width, 1), dtype=np.float32)
    for env_idx in range(num_envs):
        tile_x = env_idx % num_cols
        tile_y = env_idx // num_cols
        for y in range(tile_height):
            for x in range(tile_width):
                src_y = tile_y * tile_height + y
                src_x = tile_x * tile_width + x
                out[env_idx, y, x, 0] = tiled_np[src_y, src_x, 0]
    return out


def _reference_extract_all_uint32_tiles(
    tiled_np: np.ndarray,
    num_envs: int,
    num_cols: int,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    """NumPy reference for the uint32 (e.g. raw instance segmentation ID) case of ``extract_all_tiles_kernel``."""
    out = np.zeros((num_envs, tile_height, tile_width, 1), dtype=np.uint32)
    for env_idx in range(num_envs):
        tile_x = env_idx % num_cols
        tile_y = env_idx // num_cols
        for y in range(tile_height):
            for x in range(tile_width):
                src_y = tile_y * tile_height + y
                src_x = tile_x * tile_width + x
                out[env_idx, y, x, 0] = tiled_np[src_y, src_x, 0]
    return out


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


def _reference_extract_all_rgb_float_tiles(
    tiled_np: np.ndarray,
    num_envs: int,
    num_cols: int,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    """NumPy reference for ``extract_all_rgb_float_tiles_kernel``."""
    out = np.zeros((num_envs, tile_height, tile_width, 3), dtype=np.float32)
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
    return out


def _reference_extract_all_motion_vector_tiles(
    tiled_np: np.ndarray,
    num_envs: int,
    num_cols: int,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    """NumPy reference for the motion-vector case of ``extract_all_tiles_kernel``."""
    out = np.zeros((num_envs, tile_height, tile_width, 2), dtype=np.float32)
    for env_idx in range(num_envs):
        tile_x = env_idx % num_cols
        tile_y = env_idx // num_cols
        for y in range(tile_height):
            for x in range(tile_width):
                src_y = tile_y * tile_height + y
                src_x = tile_x * tile_width + x
                out[env_idx, y, x, 0] = tiled_np[src_y, src_x, 0]
                out[env_idx, y, x, 1] = tiled_np[src_y, src_x, 1]
    return out


class TestExtractAllMotionVectorTilesKernel:
    """Tests for the motion-vector case of ``extract_all_tiles_kernel`` used by OVRTX TargetMotionSD."""

    def test_two_by_two_tile_grid_drops_extra_channels(self):
        """The kernel reads only the first two channels, even if the source buffer has more (e.g. 4)."""
        num_cols = 2
        num_envs = 4
        tile_width = 2
        tile_height = 3
        tiled_h = (num_envs // num_cols) * tile_height
        tiled_w = num_cols * tile_width
        tiled_np = np.zeros((tiled_h, tiled_w, 4), dtype=np.float32)
        for h in range(tiled_h):
            for w in range(tiled_w):
                tiled_np[h, w, 0] = float(h * 1000 + w)
                tiled_np[h, w, 1] = float(h * 1000 + w + 100)
                tiled_np[h, w, 2] = float(h * 1000 + w + 200)
                tiled_np[h, w, 3] = float(h * 1000 + w + 300)

        tiled_wp = wp.array(tiled_np, dtype=wp.float32, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 2), dtype=wp.float32, device=DEVICE)

        wp.launch(
            kernel=extract_all_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_motion_vector_tiles(tiled_np, num_envs, num_cols, tile_width, tile_height)
        np.testing.assert_allclose(output_wp.numpy(), expected, rtol=0, atol=0)


class TestExtractAllDepthTilesKernel:
    """Tests for the depth case of ``extract_all_tiles_kernel``."""

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
            kernel=extract_all_tiles_kernel,
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
            kernel=extract_all_tiles_kernel,
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
            kernel=extract_all_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_depth_tiles(tiled_np, num_envs, num_cols, tile_width, tile_height)
        np.testing.assert_allclose(output_wp.numpy(), expected, rtol=1e-6, atol=1e-6)


class TestExtractAllUint32TilesKernel:
    """Tests for the uint32 (e.g. raw instance segmentation ID) case of ``extract_all_tiles_kernel``."""

    def test_two_by_two_tile_grid(self):
        num_cols = 2
        num_envs = 4
        tile_width = 2
        tile_height = 3
        tiled_h = (num_envs // num_cols) * tile_height
        tiled_w = num_cols * tile_width
        rng = np.random.default_rng(98765)
        tiled_np = rng.integers(0, 2**31, size=(tiled_h, tiled_w, 1), dtype=np.uint32)

        tiled_wp = wp.array(tiled_np, dtype=wp.uint32, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 1), dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=extract_all_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_uint32_tiles(tiled_np, num_envs, num_cols, tile_width, tile_height)
        np.testing.assert_array_equal(output_wp.numpy(), expected)

    def test_single_tile(self):
        num_cols = 1
        num_envs = 1
        tile_width = 4
        tile_height = 4
        tiled_np = np.arange(tile_height * tile_width, dtype=np.uint32).reshape(tile_height, tile_width, 1)

        tiled_wp = wp.array(tiled_np, dtype=wp.uint32, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 1), dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=extract_all_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_uint32_tiles(tiled_np, num_envs, num_cols, tile_width, tile_height)
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
        rng = np.random.default_rng(13579)
        tiled_np = rng.integers(0, 2**31, size=(tiled_h, tiled_w, 1), dtype=np.uint32)

        tiled_wp = wp.array(tiled_np, dtype=wp.uint32, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 1), dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=extract_all_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_uint32_tiles(tiled_np, num_envs, num_cols, tile_width, tile_height)
        np.testing.assert_array_equal(output_wp.numpy(), expected)


class TestExtractAllRgbaTilesKernel:
    """Tests for the RGB/RGBA case of ``extract_all_tiles_kernel``."""

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
            kernel=extract_all_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
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
            kernel=extract_all_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_rgba_tiles(
            tiled_np, num_envs, num_cols, tile_width, tile_height, num_channels
        )
        np.testing.assert_array_equal(output_wp.numpy(), expected)

    def test_three_channel_output_skips_alpha(self):
        """A 3-channel output buffer only copies RGB, even if the tiled input has an alpha channel."""
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
            kernel=extract_all_tiles_kernel,
            dim=(1, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_rgba_tiles(tiled_np, num_envs, num_cols, tile_width, tile_height, 3)
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
            kernel=extract_all_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_rgba_tiles(
            tiled_np, num_envs, num_cols, tile_width, tile_height, num_channels
        )
        np.testing.assert_array_equal(output_wp.numpy(), expected)


class TestExtractAllRgbFloatTilesKernel:
    """Tests for the HdrColor (float32/float16) case of ``extract_all_tiles_kernel``."""

    def test_two_by_two_tile_grid(self):
        num_cols = 2
        num_envs = 4
        tile_width = 2
        tile_height = 3
        tiled_h = (num_envs // num_cols) * tile_height
        tiled_w = num_cols * tile_width
        tiled_np = np.zeros((tiled_h, tiled_w, 3), dtype=np.float32)
        for h in range(tiled_h):
            for w in range(tiled_w):
                tiled_np[h, w, 0] = float(h * 1000 + w)
                tiled_np[h, w, 1] = float(h * 1000 + w + 100)
                tiled_np[h, w, 2] = float(h * 1000 + w + 200)

        tiled_wp = wp.array(tiled_np, dtype=wp.float32, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 3), dtype=wp.float32, device=DEVICE)

        wp.launch(
            kernel=extract_all_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_rgb_float_tiles(tiled_np, num_envs, num_cols, tile_width, tile_height)
        np.testing.assert_allclose(output_wp.numpy(), expected, rtol=0, atol=0)

    def test_half_input_writes_float_output(self):
        num_cols = 2
        num_envs = 4
        tile_width = 2
        tile_height = 3
        tiled_h = (num_envs // num_cols) * tile_height
        tiled_w = num_cols * tile_width
        tiled_np = np.zeros((tiled_h, tiled_w, 3), dtype=np.float16)
        for h in range(tiled_h):
            for w in range(tiled_w):
                tiled_np[h, w, 0] = np.float16(h * 10 + w)
                tiled_np[h, w, 1] = np.float16(h * 10 + w + 0.25)
                tiled_np[h, w, 2] = np.float16(h * 10 + w + 0.5)

        tiled_wp = wp.array(tiled_np, dtype=wp.float16, ndim=3, device=DEVICE)
        output_wp = wp.zeros(shape=(num_envs, tile_height, tile_width, 3), dtype=wp.float32, device=DEVICE)

        wp.launch(
            kernel=extract_all_tiles_kernel,
            dim=(num_envs, tile_height, tile_width),
            inputs=[tiled_wp, output_wp, num_cols, tile_width, tile_height],
            device=DEVICE,
        )
        wp.synchronize()

        expected = _reference_extract_all_rgb_float_tiles(
            tiled_np.astype(np.float32), num_envs, num_cols, tile_width, tile_height
        )
        np.testing.assert_allclose(output_wp.numpy(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")
class TestGenerateRandomColorsFromIdsKernel:
    """generate_random_colors_from_ids_kernel agrees with the shared segmentation_colors reference."""

    def _launch(self, ids_np: np.ndarray) -> np.ndarray:
        ids_wp = wp.array(ids_np, dtype=wp.uint32, device=DEVICE)
        out_wp = wp.zeros(ids_np.shape, dtype=wp.uint32, device=DEVICE)
        wp.launch(generate_random_colors_from_ids_kernel, dim=ids_np.shape, inputs=[ids_wp, out_wp], device=DEVICE)
        wp.synchronize()
        return out_wp.numpy()

    def test_matches_host_reference(self):
        """Output matches pack_rgba(random_color_from_id(id)) for a small grid of ids."""
        ids_np = np.array([[[0], [1]], [[2], [3]]], dtype=np.uint32)
        out_np = self._launch(ids_np)
        for (i, j, k), input_id in np.ndenumerate(ids_np):
            expected = pack_rgba(random_color_from_id(int(np.uint32(input_id))))
            assert int(out_np[i, j, k]) == expected, (
                f"At ({i},{j},{k}) id={input_id}: expected 0x{expected:08x}, got 0x{int(out_np[i, j, k]):08x}"
            )

    def test_deterministic_across_launches(self):
        """Two launches with the same input produce identical output."""
        rng = np.random.default_rng(42)
        ids_np = rng.integers(0, 2**31, size=(4, 4, 1), dtype=np.uint32)
        first = self._launch(ids_np).copy()
        second = self._launch(ids_np)
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("input_value", [0, 1, 2, 3, 100])
    def test_single_value(self, input_value):
        ids_np = np.array([[[input_value]]], dtype=np.uint32)
        out_np = self._launch(ids_np)
        expected = pack_rgba(random_color_from_id(int(np.uint32(input_value))))
        out_color = int(out_np[0, 0, 0])
        assert out_color == expected, (
            f"id=0x{int(np.uint32(input_value)):08x}: expected 0x{expected:08x}, got 0x{out_color:08x}"
        )
