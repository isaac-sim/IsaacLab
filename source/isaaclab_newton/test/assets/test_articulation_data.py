# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data-container tests without a simulator or asset download."""

import re

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.physics import NewtonManager
from newton.selection import ArticulationView

from isaaclab.assets.articulation.ordering import build_articulation_name_map
from isaaclab.test.utils import test_devices


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("first_property", ["body_link_jacobian_w", "mass_matrix", "gravity_compensation_forces"])
def test_task_space_buffers_are_lazy_and_reused(monkeypatch, floating, reordered, first_property):
    """Unused task-space outputs allocate nothing; first use preserves values and ordering."""
    builder = newton.ModelBuilder()
    for world in range(2):
        builder.begin_world()
        # The second articulation has more joints to exercise heterogeneous model-wide scratch.
        for name, masses in (("Robot", [4.0, 2.0, 3.0]), ("Other", [1.0] * 5)):
            links = [
                builder.add_link(mass=mass, inertia=wp.mat33(np.eye(3)), label=f"env_{world}/{name}/link_{index}")
                for index, mass in enumerate(masses)
            ]
            if floating and name == "Robot":
                joints = [builder.add_joint_free(links[0], label=f"env_{world}/{name}/root")]
            else:
                joints = [builder.add_joint_fixed(-1, links[0], label=f"env_{world}/{name}/root")]
            for index in range(1, len(links)):
                joints.append(
                    builder.add_joint_prismatic(
                        links[index - 1],
                        links[index],
                        axis=(1.0, 0.0, 0.0) if index == 1 else (0.0, 0.0, 1.0),
                        label=f"env_{world}/{name}/joint_{index}",
                    )
                )
            builder.add_articulation(joints, label=f"env_{world}/{name}")
        builder.end_world()
    model = builder.finalize(device="cpu")
    state = model.state()
    control = model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    monkeypatch.setattr(NewtonManager, "get_model", lambda: model)
    monkeypatch.setattr(NewtonManager, "get_state_0", lambda: state)
    monkeypatch.setattr(NewtonManager, "get_control", lambda: control)
    views = [
        ArticulationView(
            model, re.compile(f"env_.*/{name}"), exclude_joint_types=[newton.JointType.FREE, newton.JointType.FIXED]
        )
        for name in ("Robot", "Other")
    ]
    data, unused_data = [ArticulationData(view, "cpu") for view in views]
    joint_indices = [1, 0] if reordered else [0, 1]
    body_indices = [0, 2, 1] if reordered else [0, 1, 2]
    data.joint_ordering = build_articulation_name_map(
        kind="joint", backend_names=["x", "z"], user_names=[("x", "z")[i] for i in joint_indices], device="cpu"
    )
    data.body_ordering = build_articulation_name_map(
        kind="body",
        backend_names=["base", "x", "z"],
        user_names=[("base", "x", "z")[i] for i in body_indices],
        device="cpu",
    )
    data._apply_ordering_maps_after_resolve()

    # These are the large optional allocations, not mandatory ordering or state buffers.
    for container in (data, unused_data):
        assert container._jacobian_buf_flat is None
        assert container._mass_matrix_full_buf is None
        assert container._gravity_force_full_buf is None
    first = getattr(data, first_property)
    assert (data._jacobian_buf_flat is not None) == (first_property != "gravity_compensation_forces")
    assert (data._mass_matrix_full_buf is not None) == (first_property == "mass_matrix")
    assert (data._gravity_force_full_buf is not None) == (first_property == "gravity_compensation_forces")

    # At the coincident link origins, both prismatic axes have no angular motion.
    base_dofs = 6 if floating else 0
    jacobian = np.zeros((3, 6, base_dofs + 2), dtype=np.float32)
    if floating:
        jacobian[:, :, :6] = np.eye(6)
    jacobian[1:, 0, base_dofs] = 1.0
    jacobian[2, 2, base_dofs + 1] = 1.0
    mass = sum(
        jacobian[i].T @ np.diag([body_mass] * 3 + [1.0] * 3) @ jacobian[i]
        for i, body_mass in enumerate([4.0, 2.0, 3.0])
    )
    gravity = sum(
        jacobian[i, :3].T @ (-body_mass * model.gravity.numpy()[0]) for i, body_mass in enumerate([4.0, 2.0, 3.0])
    )
    dof_indices = list(range(base_dofs)) + [base_dofs + index for index in joint_indices]
    public_bodies = body_indices if floating else body_indices[1:]
    expected_jacobian = jacobian[public_bodies][:, :, dof_indices]
    expected_mass = mass[np.ix_(dof_indices, dof_indices)]
    expected_gravity = gravity[dof_indices]
    expected_values = {
        "body_com_jacobian_w": expected_jacobian,
        "body_link_jacobian_w": expected_jacobian,
        "mass_matrix": expected_mass,
        "gravity_compensation_forces": expected_gravity,
    }
    for name, expected in expected_values.items():
        result = getattr(data, name)
        np.testing.assert_allclose(result.warp.numpy(), np.broadcast_to(expected, result.shape), atol=1.0e-5)
        assert getattr(data, name) is result
    assert getattr(data, first_property) is first

    # Rebinding simulation arrays/order maps must not discard existing task-space output wrappers.
    data._create_simulation_bindings()
    data._apply_ordering_maps_after_resolve()
    assert getattr(data, first_property) is first
    assert unused_data._jacobian_buf_flat is None
    assert unused_data._mass_matrix_full_buf is None
    assert unused_data._gravity_force_full_buf is None


@pytest.mark.parametrize("device", test_devices())
def test_task_space_first_use_capture_replays_changed_state(monkeypatch, device):
    """First-use CUDA allocations replay correctly after a pendulum/slider changes configuration."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    base = builder.add_link(mass=4.0, inertia=wp.mat33(np.eye(3)), label="Robot/base")
    arm = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)), com=(0.5, 0.0, 0.0), label="Robot/arm")
    slider = builder.add_link(mass=3.0, inertia=wp.mat33(np.eye(3)), com=(0.25, 0.0, 0.0), label="Robot/slider")
    joints = [
        builder.add_joint_fixed(-1, base),
        builder.add_joint_revolute(base, arm, axis=(0.0, 1.0, 0.0), label="hinge"),
        builder.add_joint_prismatic(
            arm,
            slider,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()),
            label="slide",
        ),
    ]
    builder.add_articulation(joints, label="Robot")
    builder.end_world()
    model = builder.finalize(device=device)
    state, control = model.state(), model.control()
    monkeypatch.setattr(NewtonManager, "get_model", lambda: model)
    monkeypatch.setattr(NewtonManager, "get_state_0", lambda: state)
    monkeypatch.setattr(NewtonManager, "get_control", lambda: control)
    monkeypatch.setattr(NewtonManager, "forward", lambda: newton.eval_fk(model, state.joint_q, state.joint_qd, state))
    view = ArticulationView(model, "Robot", exclude_joint_types=[newton.JointType.FIXED])

    def make_data():
        data = ArticulationData(view, device)
        data.joint_ordering = build_articulation_name_map(
            kind="joint", backend_names=["hinge", "slide"], user_names=["slide", "hinge"], device=device
        )
        data.body_ordering = build_articulation_name_map(
            kind="body",
            backend_names=["base", "arm", "slider"],
            user_names=["base", "slider", "arm"],
            device=device,
        )
        data._apply_ordering_maps_after_resolve()
        data._fk_timestamp = -1.0
        return data

    properties = ("body_com_jacobian_w", "body_link_jacobian_w", "mass_matrix", "gravity_compensation_forces")
    # Compile using a separate container; the tested container must allocate for the first time inside capture.
    warm_data = make_data()
    for name in properties:
        getattr(warm_data, name)
    data = make_data()
    assert data._jacobian_buf_flat is None
    assert data._mass_matrix_full_buf is None
    assert data._gravity_force_full_buf is None
    if wp.get_device(device).is_cuda:
        assert wp.is_mempool_enabled(device), "First-use allocation during CUDA capture requires Warp mempools."
        with wp.ScopedCapture(device=device) as capture:
            outputs = {name: getattr(data, name) for name in properties}

    for angle, displacement in ((0.0, 0.0), (0.7, 0.4), (-0.3, 0.2)):
        state.joint_q.assign(np.asarray([angle, displacement], dtype=np.float32))
        if wp.get_device(device).is_cuda:
            wp.capture_launch(capture.graph)
        else:
            data._fk_timestamp = -1.0
            outputs = {name: getattr(data, name) for name in properties}
        direction = np.array([np.cos(angle), 0.0, -np.sin(angle)])
        axis = np.array([0.0, 1.0, 0.0])
        com_jacobian = np.zeros((2, 6, 2))
        for body, radius in enumerate((0.5, 1.25 + displacement)):
            com_jacobian[body, :3, 0] = np.cross(axis, radius * direction)
            com_jacobian[body, 3:, 0] = axis
        com_jacobian[1, :3, 1] = direction
        link_jacobian = com_jacobian.copy()
        link_jacobian[0, :3, 0] = 0.0
        link_jacobian[1, :3, 0] = np.cross(axis, (1.0 + displacement) * direction)
        mass = sum(
            jac.T @ np.diag([body_mass] * 3 + [1.0] * 3) @ jac for jac, body_mass in zip(com_jacobian, (2.0, 3.0))
        )
        gravity = sum(
            jac[:3].T @ (-body_mass * model.gravity.numpy()[0]) for jac, body_mass in zip(com_jacobian, (2.0, 3.0))
        )
        expected = {
            "body_com_jacobian_w": com_jacobian[::-1, :, ::-1],
            "body_link_jacobian_w": link_jacobian[::-1, :, ::-1],
            "mass_matrix": mass[::-1, ::-1],
            "gravity_compensation_forces": gravity[::-1],
        }
        for name in properties:
            np.testing.assert_allclose(outputs[name].warp.numpy()[0], expected[name], atol=1.0e-5)
            assert getattr(data, name) is outputs[name]
