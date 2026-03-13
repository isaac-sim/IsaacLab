# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX renderer kernels."""

import binascii

import numpy as np
import pytest
import warp as wp
from isaaclab_ov.renderers.ovrtx_renderer_kernels import (
    DEVICE,
    compute_crc32_hash_kernel,
)


def _crc32_reference(value: int) -> int:
    """Reference CRC32 for a single uint32 (4 bytes, little-endian)."""
    return binascii.crc32(value.to_bytes(4, "little"))


class TestComputeCrc32HashKernel:
    """Tests for compute_crc32_hash_kernel (semantic ID hashing for distinguishability)."""

    def test_crc32_matches_reference(self):
        """Kernel output matches Python reference CRC32 for known uint32 inputs."""
        inputs_np = np.array([[0, 1], [2, 3]], dtype=np.uint32)
        input_data = wp.array(inputs_np, dtype=wp.uint32, ndim=2, device=DEVICE)

        h, w = inputs_np.shape
        output_hash = wp.zeros(shape=(h, w), dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=compute_crc32_hash_kernel,
            dim=(h, w),
            inputs=[input_data, output_hash],
            device=DEVICE,
        )
        wp.synchronize()

        out_np = output_hash.numpy()
        for i in range(h):
            for j in range(w):
                input_value = inputs_np[i, j]
                ref_hash = _crc32_reference(int(input_value))
                kernel_hash = out_np[i, j]
                assert kernel_hash == ref_hash, (
                    f"At ({i},{j}): input=0x{input_value:08x}\n"
                    f"              ref_hash=0x{ref_hash:08x}\n"
                    f"              kernel_hash=0x{kernel_hash:08x}"
                )

    def test_crc32_deterministic(self):
        """Same input produces the same hash across multiple launches."""
        h, w = 4, 4
        rng = np.random.default_rng(42)
        inputs_np = rng.integers(0, 2**31, size=(h, w), dtype=np.uint32)
        input_data = wp.array(inputs_np, dtype=wp.uint32, ndim=2, device=DEVICE)
        output_hash = wp.zeros(shape=(h, w), dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=compute_crc32_hash_kernel,
            dim=(h, w),
            inputs=[input_data, output_hash],
            device=DEVICE,
        )
        wp.synchronize()
        first_run = output_hash.numpy().copy()

        wp.launch(
            kernel=compute_crc32_hash_kernel,
            dim=(h, w),
            inputs=[input_data, output_hash],
            device=DEVICE,
        )
        wp.synchronize()
        second_run = output_hash.numpy()

        np.testing.assert_array_equal(first_run, second_run)

    @pytest.mark.parametrize("input_value", [0, 1, 0x12345678, 0xFFFFFFFF])
    def test_crc32_single_value(self, input_value):
        inputs_np = np.array([[input_value]], dtype=np.uint32)
        input_data = wp.array(inputs_np, dtype=wp.uint32, ndim=2, device=DEVICE)

        h, w = inputs_np.shape
        output_hash = wp.zeros(shape=(h, w), dtype=wp.uint32, device=DEVICE)

        wp.launch(
            kernel=compute_crc32_hash_kernel,
            dim=(h, w),
            inputs=[input_data, output_hash],
            device=DEVICE,
        )
        wp.synchronize()

        ref_hash = _crc32_reference(input_value)
        kernel_hash = output_hash.numpy()[0, 0]

        assert kernel_hash == ref_hash, (
            f"val=0x{input_value:08x}: ref_hash=0x{ref_hash:08x}, kernel_hash=0x{kernel_hash:08x}"
        )
