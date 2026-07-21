# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PhysX articulation Warp kernels."""

import sys
import warnings
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_physx.assets.articulation.kernels import (
    write_joint_state_data,
    write_joint_state_data_kernel,
    write_joint_vel_data,
    write_joint_vel_data_kernel,
)


def _selector(values: list[int], dtype: type) -> wp.array:
    """Create a CPU Warp selector with the requested integer width."""
    return wp.array(values, dtype=dtype, device="cpu")


def _articulation_class():
    """Import PhysX Articulation while suppressing only unavailable lazy Kit exports."""
    cloner = ModuleType("isaaclab_physx.cloner")
    cloner.queue_physx_replication = lambda cfg: None
    physics = ModuleType("isaaclab_physx.physics")
    physics.PhysxManager = type("PhysxManager", (), {})
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "^'RigidBodyMaterialCfg' is deprecated and will be removed in 5[.]0[.] Use "
                "'isaaclab_physx[.]sim[.]spawners[.]materials[.]PhysxRigidBodyMaterialCfg' for PhysX properties, "
                "or 'isaaclab[.]sim[.]spawners[.]materials[.]RigidBodyMaterialBaseCfg' for solver-common properties "
                "only[.]$"
            ),
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=(
                "^`torch[.]jit[.]script` is deprecated[.] Please switch to `torch[.]compile` or `torch[.]export`[.]$"
            ),
            category=DeprecationWarning,
        )
        with patch.dict(sys.modules, {"isaaclab_physx.cloner": cloner, "isaaclab_physx.physics": physics}):
            from isaaclab_physx.assets.articulation.articulation import Articulation

    return Articulation


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("resolver_name", ["_resolve_env_ids", "_get_cpu_env_ids"])
def test_external_env_resolvers_return_int32_with_preserved_values(resolver_name: str, env_dtype: type) -> None:
    """Normalize every supported Warp environment selector before a PhysX boundary."""
    Articulation = _articulation_class()
    articulation = SimpleNamespace(_ALL_INDICES=_selector([0, 1], wp.int32), device="cpu")
    selector = _selector([1, 0], env_dtype)

    resolved = getattr(Articulation, resolver_name)(articulation, selector)

    assert resolved.dtype == wp.int32
    np.testing.assert_array_equal(resolved.numpy(), [1, 0])
    if resolver_name == "_resolve_env_ids" and env_dtype == wp.int32:
        assert resolved.ptr == selector.ptr


@pytest.mark.parametrize("resolver_name", ["_resolve_env_ids", "_get_cpu_env_ids"])
@pytest.mark.parametrize(
    "selector",
    [
        pytest.param(lambda: _selector([0], wp.int16), id="warp_int16"),
        pytest.param(lambda: torch.tensor([0], dtype=torch.int16), id="torch_int16"),
    ],
)
def test_external_env_resolvers_reject_unsupported_dtypes_before_boundary(resolver_name: str, selector) -> None:
    """Reject unsupported environment selector widths before calling a PhysX binding."""
    Articulation = _articulation_class()
    articulation = SimpleNamespace(_ALL_INDICES=_selector([0], wp.int32), device="cpu")

    with pytest.raises(TypeError, match="signed 32-bit or signed 64-bit integers"):
        getattr(Articulation, resolver_name)(articulation, selector())


def test_public_joint_velocity_kernel_rejects_int16_direct_launch() -> None:
    """The deprecated public raw symbol remains a concrete int32 compatibility kernel."""
    data = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    env_ids = _selector([0], wp.int16)
    joint_ids = _selector([0], wp.int32)
    outputs = [wp.zeros((1, 1), dtype=wp.float32, device="cpu") for _ in range(3)]
    with pytest.raises(RuntimeError):
        wp.launch(
            write_joint_vel_data,
            dim=(1, 1),
            inputs=[data, env_ids, joint_ids, False],
            outputs=outputs,
            device="cpu",
        )


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("joint_dtype", [wp.int32, wp.int64])
def test_write_joint_vel_data_scatters_nonidentity_selectors(env_dtype: type, joint_dtype: type) -> None:
    """Scatter compact joint velocities for every supported selector-width combination."""
    in_data = wp.array(np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32), device="cpu")
    env_ids = _selector([1, 0], env_dtype)
    joint_ids = _selector([2, 0], joint_dtype)
    joint_vel = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    prev_joint_vel = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    joint_acc = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    kernel = write_joint_vel_data
    if env_dtype != wp.int32 or joint_dtype != wp.int32:
        kernel = write_joint_vel_data_kernel(env_ids, joint_ids)

    wp.launch(
        kernel,
        dim=(2, 2),
        inputs=[in_data, env_ids, joint_ids, False],
        outputs=[joint_vel, prev_joint_vel, joint_acc],
        device="cpu",
    )

    expected_velocity = np.asarray([[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]], dtype=np.float32)
    expected_acceleration = np.asarray([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(joint_vel.numpy(), expected_velocity)
    np.testing.assert_array_equal(prev_joint_vel.numpy(), expected_velocity)
    np.testing.assert_array_equal(joint_acc.numpy(), expected_acceleration)


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("joint_dtype", [wp.int32, wp.int64])
def test_write_joint_state_data_scatters_nonidentity_selectors(env_dtype: type, joint_dtype: type) -> None:
    """Scatter compact joint state for every supported selector-width combination."""
    pos_data = wp.array(np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32), device="cpu")
    vel_data = wp.array(np.asarray([[111.0, 112.0], [121.0, 122.0]], dtype=np.float32), device="cpu")
    env_ids = _selector([1, 0], env_dtype)
    joint_ids = _selector([2, 0], joint_dtype)
    joint_pos = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    joint_vel = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    prev_joint_vel = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    joint_acc = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    kernel = write_joint_state_data
    if env_dtype != wp.int32 or joint_dtype != wp.int32:
        kernel = write_joint_state_data_kernel(env_ids, joint_ids)

    wp.launch(
        kernel,
        dim=(2, 2),
        inputs=[pos_data, vel_data, env_ids, joint_ids, False],
        outputs=[joint_pos, joint_vel, prev_joint_vel, joint_acc],
        device="cpu",
    )

    expected_position = np.asarray([[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]], dtype=np.float32)
    expected_velocity = np.asarray([[122.0, -1.0, 121.0], [112.0, -1.0, 111.0]], dtype=np.float32)
    expected_acceleration = np.asarray([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(joint_pos.numpy(), expected_position)
    np.testing.assert_array_equal(joint_vel.numpy(), expected_velocity)
    np.testing.assert_array_equal(prev_joint_vel.numpy(), expected_velocity)
    np.testing.assert_array_equal(joint_acc.numpy(), expected_acceleration)


def test_public_joint_velocity_kernel_factory_launches_int64_specialization() -> None:
    """Launch the deprecated PhysX worker's validated int64 specialization."""
    data = wp.array([[3.0]], dtype=wp.float32, device="cpu")
    env_ids = _selector([0], wp.int64)
    joint_ids = _selector([0], wp.int64)
    outputs = [wp.zeros((1, 1), dtype=wp.float32, device="cpu") for _ in range(3)]
    wp.launch(
        write_joint_vel_data_kernel(env_ids, joint_ids),
        dim=(1, 1),
        inputs=[data, env_ids, joint_ids, False],
        outputs=outputs,
        device="cpu",
    )
    assert outputs[0].numpy()[0, 0] == 3.0
