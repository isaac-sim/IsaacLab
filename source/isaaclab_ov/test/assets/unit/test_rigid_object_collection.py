# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused OVPhysX fused-layout and staging tests for rigid-object collections."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets.rigid_object_collection.rigid_object_collection import (  # noqa: E402
    RigidObjectCollection,
)
from isaaclab_ov.assets.rigid_object_collection.rigid_object_collection_data import (  # noqa: E402
    RigidObjectCollectionData,
)

from isaaclab.utils.buffers.timestamped_buffer_warp import TimestampedBufferWarp  # noqa: E402


def _collection_shell() -> RigidObjectCollection:
    """Create an N=2, B=3 collection shell with selector scratch buffers."""
    collection = object.__new__(RigidObjectCollection)
    collection._device = "cpu"
    collection._num_instances = 2
    collection._num_bodies = 3
    collection._ALL_ENV_INDICES = wp.array([0, 1], dtype=wp.int32, device="cpu")
    collection._ALL_BODY_INDICES = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
    collection._ALL_VIEW_INDICES = wp.array([0, 1, 2, 3, 4, 5], dtype=wp.int32, device="cpu")
    collection._cpu_all_view_ids = collection._ALL_VIEW_INDICES
    collection._sim_view_ids = wp.empty(6, dtype=wp.int32, device="cpu")
    collection._sim_view_ids_views = {}
    collection._cpu_view_ids = wp.empty(6, dtype=wp.int32, device="cpu")
    collection._cpu_view_ids_views = {}
    return collection


def _data_shell(device: str = "cpu") -> RigidObjectCollectionData:
    """Create an N=2, B=3 collection-data shell for pure layout helpers."""
    data = object.__new__(RigidObjectCollectionData)
    data.device = device
    data.num_instances = 2
    data.num_bodies = 3
    data._cpu_staging_buffers = {}
    return data


def test_instance_major_scalar_layout_round_trips_through_body_major_binding() -> None:
    """Scalar values must follow literal body-major fused order and invert losslessly."""
    collection = _collection_shell()
    data = _data_shell()
    public = wp.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype=wp.float32, device="cpu")

    fused = collection.reshape_data_to_view_2d(public)
    restored = data._reshape_view_to_data_2d(fused)

    np.testing.assert_array_equal(fused.numpy(), [0.0, 10.0, 1.0, 11.0, 2.0, 12.0])
    np.testing.assert_array_equal(restored.numpy(), public.numpy())


@pytest.mark.parametrize("library", ["warp", "torch"])
def test_instance_major_vector_layout_round_trips_through_body_major_binding(library: str) -> None:
    """Vector values must keep component rows attached while transposing N and B."""
    collection = _collection_shell()
    data = _data_shell()
    public_torch = torch.tensor(
        [
            [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]],
            [[10.0, 10.5], [11.0, 11.5], [12.0, 12.5]],
        ]
    )
    public = public_torch if library == "torch" else wp.from_torch(public_torch, dtype=wp.float32)

    fused = collection.reshape_data_to_view_3d(public, 2, device="cpu")
    fused_warp = wp.from_torch(fused, dtype=wp.float32) if isinstance(fused, torch.Tensor) else fused
    restored = data._reshape_view_to_data_3d(fused_warp, 2)

    expected_fused = torch.tensor([[0.0, 0.5], [10.0, 10.5], [1.0, 1.5], [11.0, 11.5], [2.0, 2.5], [12.0, 12.5]])
    torch.testing.assert_close(wp.to_torch(fused_warp), expected_fused)
    torch.testing.assert_close(wp.to_torch(restored), public_torch)


def test_env_body_selectors_map_to_literal_body_major_view_ids() -> None:
    """Nonidentity environment and body selectors must produce body-major flat IDs."""
    collection = _collection_shell()

    view_ids = collection._env_body_ids_to_view_ids(
        wp.array([1, 0], dtype=wp.int64, device="cpu"),
        wp.array([2, 0], dtype=wp.int32, device="cpu"),
        device="cpu",
    )

    np.testing.assert_array_equal(view_ids.numpy(), [5, 4, 1, 0])
    assert view_ids.dtype == wp.int32


def test_native_and_mock_binding_writes_use_their_distinct_index_domains() -> None:
    """Native fused writes use view IDs while contract mocks use environment IDs."""
    collection = _collection_shell()
    public = wp.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], dtype=wp.float32, device="cpu")
    env_ids = wp.array([1], dtype=wp.int32, device="cpu")
    calls: list[tuple[wp.array, wp.array]] = []
    collection._get_sim_env_ids = lambda ids, sim_ids=None: ids
    collection._root_view = SimpleNamespace(
        set_attribute=lambda tensor_type, values, indices=None: calls.append((values, indices))
    )

    collection._get_binding = lambda tensor_type: SimpleNamespace(shape=(6,))
    collection._binding_write(TT.BODY_MASS, public, env_ids=env_ids, device="cpu")
    native_values, native_ids = calls.pop()
    np.testing.assert_array_equal(native_values.numpy(), [0.0, 10.0, 1.0, 11.0, 2.0, 12.0])
    np.testing.assert_array_equal(native_ids.numpy(), [1, 3, 5])

    collection._get_binding = lambda tensor_type: SimpleNamespace(shape=(2, 3))
    collection._binding_write(TT.BODY_MASS, public, env_ids=env_ids, device="cpu")
    mock_values, mock_ids = calls.pop()
    assert mock_values.ptr == public.ptr
    np.testing.assert_array_equal(mock_ids.numpy(), [1])


def test_cpu_only_read_scratch_is_cached_by_tensor_type() -> None:
    """Repeated CPU-only property reads must reuse one correctly shaped staging allocation."""
    data = _data_shell()
    binding = SimpleNamespace(shape=(6, 9))

    first = data._read_view_scratch(TT.BODY_INERTIA, binding)
    second = data._read_view_scratch(TT.BODY_INERTIA, binding)

    assert first is second
    assert first.shape == (6, 9)
    assert str(first.device) == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA copy-back requires a GPU")
def test_cuda_collection_reuses_pinned_cpu_property_scratch_and_copies_instance_major() -> None:
    """A CUDA collection must read CPU properties through pinned, reusable body-major staging."""
    data = _data_shell(device="cuda:0")
    data._sim_timestamp = 1.0
    binding = SimpleNamespace(shape=(6, 1))
    body_major = wp.array([[0.0], [10.0], [1.0], [11.0], [2.0], [12.0]], dtype=wp.float32, device="cpu")

    class _View:
        def try_binding_for(self, tensor_type):
            return binding

        def read_into(self, tensor_type, destination) -> None:
            wp.copy(destination, body_major)

    data._view = _View()
    destination = TimestampedBufferWarp((2, 3), device="cuda:0", dtype=wp.float32)

    data._read_binding_into_instance_major(TT.BODY_MASS, destination, floats_per_elem=1)

    scratch = data._read_view_scratch(TT.BODY_MASS, binding)
    assert scratch is data._read_view_scratch(TT.BODY_MASS, binding)
    assert scratch.pinned
    assert str(scratch.device) == "cpu"
    torch.testing.assert_close(
        wp.to_torch(destination.data),
        torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], device="cuda:0"),
    )


def test_transform_component_views_share_storage_with_structured_parent() -> None:
    """Position and quaternion adapters must reinterpret, not copy, transform storage."""
    data = _data_shell()
    transforms = wp.array(
        [
            [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9], [4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 0.7]],
            [[7.0, 8.0, 9.0, 0.7, 0.8, 0.9, 0.1], [10.0, 11.0, 12.0, 0.2, 0.3, 0.4, 0.8]],
        ],
        dtype=wp.transformf,
        device="cpu",
    )

    positions = data._get_pos_from_transform(transforms)
    quaternions = data._get_quat_from_transform(transforms)

    assert positions.ptr == transforms.ptr
    assert quaternions.ptr == transforms.ptr + 3 * 4
    np.testing.assert_array_equal(positions.numpy()[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(quaternions.numpy()[0, 0], [0.1, 0.2, 0.3, 0.9], atol=1e-6)
