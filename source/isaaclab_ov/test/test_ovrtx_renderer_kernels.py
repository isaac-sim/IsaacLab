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
    generate_random_colors_from_ids_kernel,
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


@pytest.mark.skip(reason="OVRTX is optional and experimental feature and temporarily is excluded from testing.")
class TestRandomColorsFromIdsKernel:
    """Tests for generate_random_colors_from_ids_kernel."""

    def test_random_colors(self):
        inputs_np = np.array([[0, 1], [2, 3]], dtype=np.uint32)
        input_ids = wp.array(inputs_np, dtype=wp.uint32, ndim=2, device=DEVICE)
        h, w = inputs_np.shape
        output_colors = wp.zeros(shape=(h, w), dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel,
            dim=(h, w),
            inputs=[input_ids, output_colors],
            device=DEVICE,
        )
        wp.synchronize()

        out_np = output_colors.numpy()
        for i in range(h):
            for j in range(w):
                input_id = int(inputs_np[i, j])
                ref_color = _reference_color(input_id)
                out_color = int(out_np[i, j])
                assert out_color == ref_color, (
                    f"At ({i},{j}) id={input_id}: expected 0x{ref_color:08x}, got 0x{out_color:08x}"
                )

    def test_deterministic_across_launches(self):
        h, w = 4, 4
        rng = np.random.default_rng(42)
        inputs_np = rng.integers(0, 2**31, size=(h, w), dtype=np.uint32)
        input_ids = wp.array(inputs_np, dtype=wp.uint32, ndim=2, device=DEVICE)
        output_colors = wp.zeros(shape=(h, w), dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel,
            dim=(h, w),
            inputs=[input_ids, output_colors],
            device=DEVICE,
        )
        wp.synchronize()
        first_run = output_colors.numpy().copy()

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel,
            dim=(h, w),
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
            kernel=generate_random_colors_from_ids_kernel,
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
