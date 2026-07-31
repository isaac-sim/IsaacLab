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
import warp as wp
from isaaclab_physx.assets.articulation.kernels import (
    write_joint_state_data,
    write_joint_state_data_kernel,
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
