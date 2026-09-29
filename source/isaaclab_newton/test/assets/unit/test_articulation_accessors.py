# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Newton articulation accessors checked on a directly built model: lazy task-space allocation with graph capture,
and the public body order of the collision-shape counts.
"""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import JointType, ModelBuilder
from newton.selection import ArticulationView

from isaaclab.test.utils import test_devices

pytestmark = pytest.mark.unit


def test_num_shapes_per_body_follows_public_body_order() -> None:
    """Align Newton shape counts with the public body-name axis."""

    class _ShapeCountSurface:
        backend_num_shapes_per_body = Articulation.backend_num_shapes_per_body
        num_shapes_per_body = Articulation.num_shapes_per_body

    articulation = _ShapeCountSurface()
    articulation._num_shapes_per_body_backend = None
    articulation._root_view = SimpleNamespace(
        body_shapes=((), (object(), object()), (object(), object(), object())),
    )
    articulation.body_ordering = SimpleNamespace(
        user_to_backend_indices=(2, 0, 1),
    )

    assert articulation.num_shapes_per_body == [3, 0, 2]


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize(
    "first_property",
    ["body_com_jacobian_w", "body_link_jacobian_w", "mass_matrix", "gravity_compensation_forces"],
)
def test_task_space_allocation_and_capture(monkeypatch, device, first_property):
    """Allocate only requested outputs and retain them across changed-state graph replay."""
    builder = ModelBuilder()
    builder.begin_world()
    base = builder.add_link(mass=4.0, inertia=wp.mat33(np.eye(3)), label="Robot/base")
    arm = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)), com=(0.5, 0.0, 0.0), label="Robot/arm")
    slider = builder.add_link(mass=3.0, inertia=wp.mat33(np.eye(3)), com=(0.25, 0.0, 0.0), label="Robot/slider")
    builder.add_articulation(
        [
            builder.add_joint_fixed(-1, base),
            builder.add_joint_revolute(base, arm, axis=(0.0, 1.0, 0.0), label="hinge"),
            builder.add_joint_prismatic(arm, slider, axis=(1.0, 0.0, 0.0), label="slide"),
        ],
        label="Robot",
    )
    builder.end_world()
    model = builder.finalize(device=device)
    state, control = model.state(), model.control()
    monkeypatch.setattr(SimulationManager, "get_model", lambda: model)
    monkeypatch.setattr(SimulationManager, "get_state_0", lambda: state)
    monkeypatch.setattr(SimulationManager, "get_control", lambda: control)
    view = ArticulationView(model, "Robot", exclude_joint_types=[JointType.FIXED])
    eager = ArticulationData(view, device)
    eager._apply_ordering_maps_after_resolve()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    getattr(eager, first_property)  # Compile kernels before first-use capture on a fresh container.
    data = ArticulationData(view, device)
    data._apply_ordering_maps_after_resolve()
    properties = ("body_com_jacobian_w", "body_link_jacobian_w", "mass_matrix", "gravity_compensation_forces")
    assert all(getattr(data, f"_{name}_ta") is None for name in properties)
    assert data._jacobian_buf_flat is data._mass_matrix_full_buf is data._gravity_force_full_buf is None
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            output = getattr(data, first_property)
    else:
        output = getattr(data, first_property)
    required = {first_property}
    if first_property == "body_link_jacobian_w":
        required.add("body_com_jacobian_w")
    for name in properties:
        assert (getattr(data, f"_{name}_ta") is not None) == (name in required)
    assert (data._jacobian_buf_flat is not None) == (first_property != "gravity_compensation_forces")
    assert (data._mass_matrix_full_buf is not None) == (first_property == "mass_matrix")
    assert (data._gravity_force_full_buf is not None) == (first_property == "gravity_compensation_forces")

    for angle, displacement in ((0.0, 0.0), (0.7, 0.4), (-0.3, 0.2)):
        state.joint_q.assign(np.asarray([angle, displacement], dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        expected = getattr(eager, first_property).warp.numpy()
        if wp.get_device(device).is_cuda:
            wp.capture_launch(capture.graph)
        else:
            assert getattr(data, first_property) is output
        np.testing.assert_allclose(output.warp.numpy(), expected, atol=1e-5)
    data._create_simulation_bindings()
    data._apply_ordering_maps_after_resolve()
    assert getattr(data, first_property) is output
