# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused joint-property staging tests for Newton articulations."""

from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import ModelBuilder, ModelFlags

from isaaclab.utils.warp.proxy_array import ProxyArray


def test_partial_joint_property_stages_user_and_backend_order_and_notifies(monkeypatch) -> None:
    """A partial public-order write must scatter to Newton order and emit one exact notification."""
    articulation = object.__new__(Articulation)
    articulation._device = "cpu"
    articulation._check_shapes = False
    articulation._ALL_INDICES = wp.array([0, 1], dtype=wp.int32, device="cpu")
    articulation._ALL_JOINT_INDICES = wp.array([0, 1], dtype=wp.int32, device="cpu")
    user_to_backend = wp.array([1, 0], dtype=wp.int32, device="cpu")
    backend_to_user = wp.array([1, 0], dtype=wp.int32, device="cpu")
    user_stiffness = wp.full((2, 2), 3.0, dtype=wp.float32, device="cpu")
    backend_stiffness = wp.full((2, 2), 3.0, dtype=wp.float32, device="cpu")
    articulation._data = SimpleNamespace(
        has_joint_ordering=True,
        joint_ordering=SimpleNamespace(user_to_backend=user_to_backend, backend_to_user=backend_to_user),
        _joint_stiffness_user=user_stiffness,
        _sim_bind_joint_stiffness_sim=backend_stiffness,
    )
    notifications = []
    monkeypatch.setattr(
        SimulationManager,
        "add_model_change",
        classmethod(lambda cls, flag: notifications.append(flag)),
    )

    articulation.write_joint_stiffness_to_sim_index(
        stiffness=wp.array([[17.0]], dtype=wp.float32, device="cpu"),
        env_ids=wp.array([1], dtype=wp.int32, device="cpu"),
        joint_ids=wp.array([0], dtype=wp.int32, device="cpu"),
    )

    np.testing.assert_allclose(user_stiffness.numpy(), [[3.0, 3.0], [17.0, 3.0]])
    np.testing.assert_allclose(backend_stiffness.numpy(), [[3.0, 3.0], [3.0, 17.0]])
    assert notifications == [ModelFlags.JOINT_DOF_PROPERTIES]


def test_num_shapes_per_body_follows_public_body_order() -> None:
    """Newton collision-shape counts must use the same public axis as body names."""

    class _ShapeCountSurface:
        backend_num_shapes_per_body = Articulation.backend_num_shapes_per_body
        num_shapes_per_body = Articulation.num_shapes_per_body

    articulation = _ShapeCountSurface()
    articulation._num_shapes_per_body_backend = None
    articulation._root_view = SimpleNamespace(body_shapes=((), (object(), object()), (object(), object(), object())))
    articulation.body_ordering = SimpleNamespace(user_to_backend_indices=(2, 0, 1))

    assert articulation.num_shapes_per_body == [3, 0, 2]


def test_viscous_writer_updates_finalized_newton_binding_and_notifies(monkeypatch) -> None:
    """Passive damping must reach Newton's live field without changing actuator derivative gains."""
    builder = ModelBuilder()
    link = builder.add_link(mass=1.0, inertia=wp.mat33(1.0))
    joint = builder.add_joint_revolute(-1, link, label="joint")
    builder.add_articulation([joint], label="articulation")
    model = builder.finalize(device="cpu")
    model_damping = wp.array(
        ptr=model.joint_damping.ptr,
        dtype=wp.float32,
        shape=(1, 1),
        strides=(model.joint_damping.strides[0], model.joint_damping.strides[0]),
        device="cpu",
        copy=False,
    )
    data_type = type("_Data", (), {"joint_viscous_friction_coeff": ArticulationData.joint_viscous_friction_coeff})
    data = data_type()
    data.has_joint_ordering = False
    data.joint_ordering = None
    data._joint_viscous_friction_user = None
    data._sim_bind_joint_viscous_friction_coeff = model_damping
    data._joint_viscous_friction_coeff_ta = ProxyArray(model_damping)
    articulation = object.__new__(Articulation)
    articulation._device = "cpu"
    articulation._check_shapes = False
    articulation._data = data
    articulation._root_view = SimpleNamespace(count=1)
    articulation._ALL_INDICES = wp.array([0], dtype=wp.int32, device="cpu")
    articulation._ALL_JOINT_INDICES = wp.array([0], dtype=wp.int32, device="cpu")
    notifications = []
    monkeypatch.setattr(
        SimulationManager,
        "add_model_change",
        classmethod(lambda cls, flag: notifications.append(flag)),
    )

    articulation.write_joint_viscous_friction_coefficient_to_sim_index(
        joint_viscous_friction_coeff=torch.tensor([[0.25]], dtype=torch.float32),
    )

    torch.testing.assert_close(data.joint_viscous_friction_coeff.torch, torch.tensor([[0.25]]))
    torch.testing.assert_close(torch.from_numpy(model.joint_damping.numpy()), torch.tensor([0.25]))
    assert notifications == [ModelFlags.JOINT_DOF_PROPERTIES]
