# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for mock OVPhysX tensor bindings."""

import numpy as np
import pytest
import warp as wp

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ovphysx.test.mock_interfaces import MockTensorBinding  # noqa: E402


def _make_binding(data: np.ndarray) -> MockTensorBinding:
    """Create a mock binding containing deterministic data."""
    binding = MockTensorBinding(tensor_type=0, shape=data.shape, count=data.shape[0])
    binding.write(data)
    return binding


def test_read_copies_into_numpy_destination():
    source = np.arange(14, dtype=np.float32).reshape(2, 7)
    destination = np.zeros_like(source)

    _make_binding(source).read(destination)

    np.testing.assert_array_equal(destination, source)


def test_read_copies_into_flat_warp_destination():
    source = np.arange(14, dtype=np.float32).reshape(2, 7)
    destination = wp.zeros(source.shape, dtype=wp.float32, device="cpu")

    _make_binding(source).read(destination)

    np.testing.assert_array_equal(destination.numpy(), source)


@pytest.mark.parametrize(("dtype", "width"), [(wp.transformf, 7), (wp.spatial_vectorf, 6)])
def test_read_copies_into_structured_warp_destination(dtype, width):
    source = np.arange(2 * width, dtype=np.float32).reshape(2, width)
    destination = wp.zeros((2,), dtype=dtype, device="cpu")

    _make_binding(source).read(destination)

    np.testing.assert_array_equal(destination.view(wp.float32).numpy(), source)


@pytest.mark.parametrize(
    ("dtype", "destination_shape"),
    [(wp.int32, (2, 3)), (wp.uint32, (2, 3)), (wp.vec3i, (2,))],
)
def test_read_rejects_warp_destination_without_float32_scalar(dtype, destination_shape):
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    destination = wp.zeros(destination_shape, dtype=dtype, device="cpu")

    with pytest.raises(RuntimeError):
        _make_binding(source).read(destination)
