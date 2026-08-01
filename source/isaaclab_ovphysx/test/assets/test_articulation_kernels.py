# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVPhysX articulation Warp kernels."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_ovphysx.assets import kernels


def _selector(values: list[int], dtype: type) -> wp.array:
    return wp.array(values, dtype=dtype, device="cpu")


def _articulation_class():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "^'RigidBodyMaterialCfg' is deprecated and will be removed in 5\\.0\\. Use "
                "'isaaclab_physx\\.sim\\.spawners\\.materials\\.PhysxRigidBodyMaterialCfg' for PhysX properties, "
                "or 'isaaclab\\.sim\\.spawners\\.materials\\.RigidBodyMaterialBaseCfg' for solver-common properties "
                "only\\.$"
            ),
            category=DeprecationWarning,
        )
        from isaaclab_ovphysx.assets.articulation.articulation import Articulation

    return Articulation


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
def test_root_worker_accepts_selector_widths(env_dtype: type) -> None:
    env_ids = _selector([1, 0], env_dtype)
    data = wp.array(
        [[11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=wp.transformf,
        device="cpu",
    )
    output = wp.zeros(2, dtype=wp.transformf, device="cpu")
    sim_env_ids = wp.empty(2, dtype=wp.int32, device="cpu")
    kernel = kernels.set_root_link_pose_to_sim_index
    if env_dtype == wp.int64:
        kernel = kernels.set_root_link_pose_to_sim_index_kernel(env_ids)

    wp.launch(kernel, dim=2, inputs=[data, env_ids], outputs=[output, sim_env_ids], device="cpu")

    np.testing.assert_array_equal(output.numpy(), data.numpy()[[1, 0]])
    np.testing.assert_array_equal(sim_env_ids.numpy(), [1, 0])


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("item_dtype", [wp.int32, wp.int64])
def test_item_worker_accepts_selector_widths(env_dtype: type, item_dtype: type) -> None:
    env_ids = _selector([1, 0], env_dtype)
    item_ids = _selector([2, 0], item_dtype)
    data = wp.array([[11.0, 12.0], [21.0, 22.0]], dtype=wp.float32, device="cpu")
    output = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    kernel = kernels.write_2d_data_to_buffer_with_indices
    if env_dtype != wp.int32 or item_dtype != wp.int32:
        kernel = kernels.write_2d_data_to_buffer_with_indices_kernel(env_ids, item_ids)

    wp.launch(kernel, dim=(2, 2), inputs=[data, env_ids, item_ids], outputs=[output], device="cpu")

    np.testing.assert_array_equal(output.numpy(), [[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]])


def test_cpu_env_ids_all_returns_pinned_fast_path() -> None:
    Articulation = _articulation_class()
    all_indices = _selector([0, 1], wp.int32)
    cpu_env_ids_all = wp.clone(all_indices, device="cpu")
    articulation = SimpleNamespace(
        _ALL_INDICES=all_indices,
        _cpu_env_ids_all=cpu_env_ids_all,
        _device="cpu",
    )

    resolved = Articulation._get_cpu_env_ids(articulation, all_indices)

    assert resolved is cpu_env_ids_all
