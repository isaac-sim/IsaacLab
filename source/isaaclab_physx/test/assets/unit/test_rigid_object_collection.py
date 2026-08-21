# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused PhysX rigid-object-collection ordering and selector tests."""

import numpy as np
import torch
import warp as wp
from isaaclab_physx.assets.rigid_object_collection.kernels import resolve_view_ids, resolve_view_ids_kernel

from ._imports import import_physx_module


def test_view_id_kernel_maps_instance_body_grid_to_physx_body_major_order() -> None:
    """Nontrivial selectors must produce literal PhysX body-major flat indices."""
    env_ids = wp.array([2, 0], dtype=wp.int64, device="cpu")
    body_ids = wp.array([1, 0], dtype=wp.int32, device="cpu")
    out = wp.empty(4, dtype=wp.int32, device="cpu")

    wp.launch(
        resolve_view_ids_kernel(env_ids, body_ids),
        dim=(2, 2),
        inputs=[env_ids, body_ids, 2, 3],
        outputs=[out],
        device="cpu",
    )

    np.testing.assert_array_equal(out.numpy(), [5, 3, 2, 0])


def test_view_id_kernel_supports_canonical_int32_selectors() -> None:
    """Canonical selectors must use the supported static kernel with the same ordering."""
    env_ids = wp.array([1, 0], dtype=wp.int32, device="cpu")
    body_ids = wp.array([2], dtype=wp.int32, device="cpu")
    out = wp.empty(2, dtype=wp.int32, device="cpu")

    wp.launch(resolve_view_ids, dim=(2, 1), inputs=[env_ids, body_ids, 2, 3], outputs=[out], device="cpu")

    np.testing.assert_array_equal(out.numpy(), [7, 6])


def test_collection_data_reshapes_physx_body_major_rows_to_instance_major() -> None:
    """PhysX body-major property rows must become public (instance, body, component) order."""
    module = import_physx_module("isaaclab_physx.assets.rigid_object_collection.rigid_object_collection_data")
    data = object.__new__(module.RigidObjectCollectionData)
    data.num_instances = 3
    data.num_bodies = 2
    data.device = "cpu"
    raw = wp.array(
        np.asarray(
            [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0], [100.0, 101.0], [110.0, 111.0], [120.0, 121.0]],
            dtype=np.float32,
        ),
        device="cpu",
    )

    result = data._reshape_view_to_data_3d(raw, 2)

    np.testing.assert_array_equal(
        result.numpy(),
        np.asarray(
            [
                [[0.0, 1.0], [100.0, 101.0]],
                [[10.0, 11.0], [110.0, 111.0]],
                [[20.0, 21.0], [120.0, 121.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_collection_view_id_conversion_stages_partial_cuda_query_on_cpu(monkeypatch) -> None:
    """A partial CUDA selector must return synchronized CPU indices accepted by the TensorAPI."""
    module = import_physx_module("isaaclab_physx.assets.rigid_object_collection.rigid_object_collection")
    collection = object.__new__(module.RigidObjectCollection)
    collection._device = "cuda:0"
    collection._root_view = type("View", (), {"count": 6})()
    collection._body_names_list = ["left", "right"]
    collection._ALL_ENV_INDICES = wp.array([0, 1, 2], dtype=wp.int32, device="cuda:0")
    collection._ALL_BODY_INDICES = wp.array([0, 1], dtype=wp.int32, device="cuda:0")
    collection._ALL_VIEW_INDICES = wp.array([0, 1, 2, 3, 4, 5], dtype=wp.int32, device="cuda:0")
    collection._cpu_all_view_ids = wp.array([0, 1, 2, 3, 4, 5], dtype=wp.int32, device="cpu")
    collection._sim_view_ids = wp.empty(6, dtype=wp.int32, device="cuda:0")
    collection._cpu_view_ids = wp.empty(6, dtype=wp.int32, device="cpu", pinned=True)
    collection._sim_view_ids_views = {}
    collection._cpu_view_ids_views = {}
    synchronize_calls = []
    monkeypatch.setattr(wp, "synchronize_stream", lambda device: synchronize_calls.append(device))

    result = collection._env_body_ids_to_view_ids(
        torch.tensor([2, 0], device="cuda:0"), torch.tensor([1], device="cuda:0"), device="cpu"
    )

    np.testing.assert_array_equal(result.numpy(), [5, 3])
    assert str(result.device) == "cpu"
    assert synchronize_calls == ["cuda:0"]
