# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused PhysX articulation staging, cache, friction, and kernel tests."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import warp as wp
from isaaclab_physx.assets.articulation.kernels import (
    extract_friction_properties,
    write_joint_friction_data_to_buffer,
    write_joint_state_data,
    write_joint_state_data_kernel,
)

from ._imports import import_physx_module


def _selector(values: list[int], dtype: type) -> wp.array:
    """Create a CPU Warp selector with the requested integer width."""
    return wp.array(values, dtype=dtype, device="cpu")


def _articulation_class():
    """Import PhysX Articulation while suppressing only unavailable lazy Kit exports."""
    return import_physx_module("isaaclab_physx.assets.articulation.articulation").Articulation


def _model_writer_articulation(Articulation) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Create a minimal receiver for PhysX model-property writer methods."""
    ArticulationData = Articulation.__init__.__globals__["ArticulationData"]
    env_ids = _selector([0], wp.int32)
    joint_ids = _selector([0], wp.int32)
    body_ids = _selector([0], wp.int32)
    data = SimpleNamespace(
        _sim_timestamp=3.0,
        _joint_armature=wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
        _joint_armature_backend=None,
        _body_mass=SimpleNamespace(data=wp.zeros((1, 1), dtype=wp.float32, device="cpu"), timestamp=3.0),
        _body_mass_backend=None,
        _body_inertia=SimpleNamespace(data=wp.zeros((1, 1, 9), dtype=wp.float32, device="cpu"), timestamp=3.0),
        _body_inertia_backend=None,
        _body_com_jacobian_w=SimpleNamespace(timestamp=3.0),
        _mass_matrix=SimpleNamespace(timestamp=3.0),
        _gravity_compensation_forces=SimpleNamespace(timestamp=3.0),
        has_body_ordering=False,
    )
    data._reset_dynamics = lambda **kwargs: ArticulationData._reset_dynamics(data, **kwargs)
    articulation = SimpleNamespace(
        data=data,
        device="cpu",
        num_instances=1,
        num_joints=1,
        num_bodies=1,
        root_view=SimpleNamespace(
            set_dof_armatures=lambda *args, **kwargs: None,
            set_masses=lambda *args, **kwargs: None,
            set_inertias=lambda *args, **kwargs: None,
        ),
        _resolve_env_ids=lambda ids: env_ids,
        _resolve_joint_ids=lambda ids: joint_ids,
        _resolve_body_ids=lambda ids: body_ids,
        _sim_env_ids_view=lambda count: env_ids,
        _get_cpu_env_ids=lambda ids, sim_ids=None: env_ids,
        _get_backend_ordered_joint_buffer=lambda user, backend: user,
        _body_user_to_backend_map=lambda: body_ids,
        assert_shape_and_dtype=lambda *args, **kwargs: None,
    )
    return articulation, data


@pytest.mark.parametrize(
    ("writer_name", "kwargs", "invalidates_gravity"),
    [
        ("write_joint_armature_to_sim_index", {"armature": 1.0}, False),
        ("set_masses_index", {"masses": wp.ones((1, 1), dtype=wp.float32, device="cpu")}, True),
        ("set_inertias_index", {"inertias": wp.ones((1, 1, 9), dtype=wp.float32, device="cpu")}, False),
    ],
)
def test_model_property_writers_invalidate_same_timestamp_dynamics(
    writer_name: str, kwargs: dict, invalidates_gravity: bool
) -> None:
    """Invalidate every computed-dynamics cache affected by a PhysX model write."""
    Articulation = _articulation_class()
    articulation, data = _model_writer_articulation(Articulation)
    method = getattr(Articulation, writer_name)
    globals_ = method.__globals__

    with (
        patch.object(globals_["wp"], "launch"),
        patch.object(globals_["ordering_kernels"], "write_float_user_to_backend_with_indices_and_sim_ids"),
        patch.object(globals_["ordering_kernels"], "write_3d_user_to_backend_with_indices_and_sim_ids"),
    ):
        method(articulation, **kwargs)

    assert data._mass_matrix.timestamp == -1.0
    expected_gravity_timestamp = -1.0 if invalidates_gravity else 3.0
    assert data._gravity_compensation_forces.timestamp == expected_gravity_timestamp
    assert data._body_com_jacobian_w.timestamp == 3.0


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


def test_friction_mapping_keeps_static_dynamic_and_viscous_components_distinct() -> None:
    """The PhysX friction tuple must map to the three public coefficient buffers without aliasing."""
    source = wp.array(
        np.asarray([[[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]], dtype=np.float32),
        device="cpu",
    )
    static = wp.empty((1, 2), dtype=wp.float32, device="cpu")
    dynamic = wp.empty_like(static)
    viscous = wp.empty_like(static)

    wp.launch(
        extract_friction_properties,
        dim=(1, 2),
        inputs=[source],
        outputs=[static, dynamic, viscous],
        device="cpu",
    )

    np.testing.assert_array_equal(static.numpy(), np.asarray([[0.1, 1.1]], dtype=np.float32))
    np.testing.assert_array_equal(dynamic.numpy(), np.asarray([[0.2, 1.2]], dtype=np.float32))
    np.testing.assert_array_equal(viscous.numpy(), np.asarray([[0.3, 1.3]], dtype=np.float32))


def test_friction_writer_preserves_components_and_nonidentity_selection() -> None:
    """Write all three coefficients to their literal TensorAPI slots for selected joints only."""
    static_in = wp.array(np.asarray([[0.1, 0.2], [1.1, 1.2]], dtype=np.float32), device="cpu")
    dynamic_in = wp.array(np.asarray([[0.3, 0.4], [1.3, 1.4]], dtype=np.float32), device="cpu")
    viscous_in = wp.array(np.asarray([[0.5, 0.6], [1.5, 1.6]], dtype=np.float32), device="cpu")
    env_ids = _selector([1, 0], wp.int32)
    joint_ids = _selector([2, 0], wp.int32)
    static = wp.full((2, 3), -1.0, dtype=wp.float32, device="cpu")
    dynamic = wp.full_like(static, -1.0)
    viscous = wp.full_like(static, -1.0)
    properties = wp.full((2, 3, 3), -1.0, dtype=wp.float32, device="cpu")
    sim_env_ids = wp.full((2,), -1, dtype=wp.int32, device="cpu")

    wp.launch(
        write_joint_friction_data_to_buffer,
        dim=(2, 2),
        inputs=[static_in, dynamic_in, viscous_in, env_ids, joint_ids, False],
        outputs=[static, dynamic, viscous, properties, sim_env_ids],
        device="cpu",
    )

    expected_properties = np.full((2, 3, 3), -1.0, dtype=np.float32)
    expected_properties[1, 2] = [0.1, 0.3, 0.5]
    expected_properties[1, 0] = [0.2, 0.4, 0.6]
    expected_properties[0, 2] = [1.1, 1.3, 1.5]
    expected_properties[0, 0] = [1.2, 1.4, 1.6]
    np.testing.assert_array_equal(properties.numpy(), expected_properties)
    np.testing.assert_array_equal(sim_env_ids.numpy(), [1, 0])


def test_int64_sim_selector_is_narrowed_on_the_articulation_device() -> None:
    """PhysX TensorAPI selectors must be int32 even when the public selector is int64."""
    Articulation = _articulation_class()
    articulation = object.__new__(Articulation)
    articulation._device = "cpu"
    env_ids = wp.array([2, 0], dtype=wp.int64, device="cpu")

    result = articulation._get_sim_env_ids(env_ids)

    assert result.dtype == wp.int32
    np.testing.assert_array_equal(result.numpy(), [2, 0])


def test_joint_property_3d_buffer_is_reordered_from_backend_to_public_order() -> None:
    """A backend 3-D joint buffer must expose the configured public joint order exactly once."""
    Articulation = _articulation_class()
    articulation = object.__new__(Articulation)
    articulation._device = "cpu"
    articulation._root_view = SimpleNamespace(count=1, shared_metatype=SimpleNamespace(dof_count=3))
    articulation._data = SimpleNamespace(
        has_joint_ordering=True,
        joint_ordering=SimpleNamespace(user_to_backend=wp.array([2, 0, 1], dtype=wp.int32, device="cpu")),
    )
    backend = wp.array(
        np.asarray([[[20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]], dtype=np.float32),
        device="cpu",
    )
    public = wp.zeros_like(backend)

    result = articulation._get_user_ordered_joint_3d_buffer(backend, public, 2)

    assert result is public
    np.testing.assert_array_equal(public.numpy(), [[[40.0, 41.0], [20.0, 21.0], [30.0, 31.0]]])
