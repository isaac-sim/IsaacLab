# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless tests for the Newton joint coordinate/DOF conversion."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.assets.articulation.joint_coordinates import (
    BallJointCoordinateMap,
    build_ball_joint_coordinate_map,
    gather_joint_coordinates,
    scatter_joint_coordinates,
)
from isaaclab_newton.physics import NewtonManager
from newton.selection import ArticulationView
from scipy.spatial.transform import Rotation

# One revolute, one ball, one revolute -- the layout that hides an off-by-one when the tables are
# built by walking the model instead of the view's own per-joint counts.
COORD_COUNTS = [1, 4, 1]
DOF_COUNTS = [1, 3, 1]

# Two balls, straddling a revolute on each side -- the second ball's offsets (ball_coord[1] = 5,
# ball_dof[1] = 4) are not exercised by the single-ball layout above.
TWO_BALL_COORD_COUNTS = [1, 4, 4, 1]
TWO_BALL_DOF_COUNTS = [1, 3, 3, 1]


def _map() -> BallJointCoordinateMap:
    return build_ball_joint_coordinate_map(COORD_COUNTS, DOF_COUNTS, "cpu")


def test_two_ball_tables_cover_every_dof() -> None:
    """A second ball joint's offsets are not a simple repeat of the first's."""
    m = build_ball_joint_coordinate_map(TWO_BALL_COORD_COUNTS, TWO_BALL_DOF_COUNTS, "cpu")
    assert list(m.ball_dof.numpy()) == [1, 4]
    assert list(m.ball_coord.numpy()) == [1, 5]
    covered = list(m.single_dof.numpy()) + [b + k for b in m.ball_dof.numpy() for k in range(3)]
    assert sorted(covered) == list(range(sum(TWO_BALL_DOF_COUNTS)))
    assert sorted(list(m.single_coord.numpy()) + [b + k for b in m.ball_coord.numpy() for k in range(4)]) == list(
        range(sum(TWO_BALL_COORD_COUNTS))
    )


def test_zero_rotation_vector_scatters_to_identity_quaternion() -> None:
    """The ``angle > 1e-9`` guard's else-branch -- taken on every reset of a passive ball joint at
    ``default_joint_pos = 0`` -- must produce the identity quaternion, not ``0/0``."""
    m = _map()
    n_dofs, n_coords = sum(DOF_COUNTS), sum(COORD_COUNTS)
    dofs = wp.zeros((1, n_dofs), dtype=wp.float32, device="cpu")
    coords = wp.zeros((1, n_coords), dtype=wp.float32, device="cpu")
    mask = wp.array(np.array([True]), dtype=wp.bool, device="cpu")

    scatter_joint_coordinates(m, dofs, coords, mask)

    ball_coord = int(m.ball_coord.numpy()[0])
    np.testing.assert_array_equal(coords.numpy()[0, ball_coord : ball_coord + 4], [0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_identity_quaternion_gathers_to_zero_rotation_vector(sign: float) -> None:
    """The inverse of the guard above: the identity quaternion (either hemisphere) decodes to the
    zero rotation vector, not a NaN from a degenerate axis normalization."""
    m = _map()
    n_dofs, n_coords = sum(DOF_COUNTS), sum(COORD_COUNTS)
    coords_np = np.zeros((1, n_coords), dtype=np.float32)
    ball_coord = int(m.ball_coord.numpy()[0])
    coords_np[0, ball_coord : ball_coord + 4] = np.array([0.0, 0.0, 0.0, 1.0]) * sign
    out = wp.zeros((1, n_dofs), dtype=wp.float32, device="cpu")

    gather_joint_coordinates(m, wp.array(coords_np, dtype=wp.float32, device="cpu"), out)

    ball_dof = int(m.ball_dof.numpy()[0])
    np.testing.assert_array_equal(out.numpy()[0, ball_dof : ball_dof + 3], [0.0, 0.0, 0.0])


@pytest.mark.parametrize("ball_joint", [False, True])
def test_joint_commands_preserve_coordinate_layout(monkeypatch, ball_joint):
    """DOF commands reach native coordinate targets without touching either world's free root."""
    source = newton.ModelBuilder()
    base, middle, tip = [source.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(3)]
    add_joint = source.add_joint_ball if ball_joint else source.add_joint_revolute
    source.add_articulation(
        [source.add_joint_free(base), add_joint(base, middle), source.add_joint_revolute(middle, tip)], label="Robot"
    )
    builder = newton.ModelBuilder()
    for _ in range(2):
        builder.add_world(source)
    model = builder.finalize(device="cpu")
    state, control = model.state(), model.control()
    assert model.joint_coord_count != model.joint_dof_count
    assert control.joint_target_q.shape == state.joint_q.shape
    monkeypatch.setattr(NewtonManager, "get_model", lambda: model)
    monkeypatch.setattr(NewtonManager, "get_state_0", lambda: state)
    monkeypatch.setattr(NewtonManager, "get_control", lambda: control)
    view = ArticulationView(model, "Robot", exclude_joint_types=[newton.JointType.FREE])
    data = ArticulationData(view, "cpu")
    data._apply_ordering_maps_after_resolve()
    num_dofs = 4 if ball_joint else 2
    assert data.joint_pos.shape == (2, num_dofs)

    targets = np.arange(2 * num_dofs, dtype=np.float32).reshape(2, num_dofs) * 0.1
    expected = control.joint_target_q.numpy().reshape(2, -1).copy()
    if ball_joint:
        expected[:, 7:-1] = Rotation.from_rotvec(targets[:, :3]).as_quat()
        expected[:, -1] = targets[:, -1]
    else:
        expected[:, 7:] = targets
    collection = SimpleNamespace(
        has_implicit_actuators=True,
        _joint_pos_target_sim=wp.array(targets, device="cpu"),
        _joint_vel_target_sim=wp.zeros((2, num_dofs), device="cpu"),
        _joint_effort_target_sim=wp.zeros((2, num_dofs), device="cpu"),
    )
    articulation = SimpleNamespace(data=data, _ALL_ENV_MASK=wp.array([True, True], dtype=wp.bool, device="cpu"))
    NewtonActuatorControl(articulation).submit_commands(collection)
    np.testing.assert_allclose(control.joint_target_q.numpy().reshape(2, -1), expected, atol=1e-6)


def test_map_is_inert_without_ball_joints() -> None:
    """An articulation whose joints all have one coordinate per DOF needs no conversion."""
    assert not build_ball_joint_coordinate_map([1, 1, 1], [1, 1, 1], "cpu").required
    assert not build_ball_joint_coordinate_map([], [], "cpu").required


def test_unsupported_layout_is_rejected() -> None:
    """A distance joint (7 coordinates, 6 DOFs) must not be decoded as a quaternion."""
    with pytest.raises(NotImplementedError, match="7 coordinates against 6 DOFs"):
        build_ball_joint_coordinate_map([7], [6], "cpu")


def test_scatter_then_gather_round_trips() -> None:
    """DOF values survive a trip through coordinate space."""
    num_envs = 2
    m = _map()
    n_dofs, n_coords = sum(DOF_COUNTS), sum(COORD_COUNTS)
    rng = np.random.default_rng(0)
    dofs_np = rng.uniform(-0.7, 0.7, size=(num_envs, n_dofs)).astype(np.float32)
    dofs = wp.array(dofs_np, dtype=wp.float32, device="cpu")
    coords = wp.zeros((num_envs, n_coords), dtype=wp.float32, device="cpu")
    mask = wp.array(np.ones(num_envs, dtype=bool), dtype=wp.bool, device="cpu")

    scatter_joint_coordinates(m, dofs, coords, mask)
    # The quaternion the scatter wrote must match an independent exp map.
    ball_coord = int(m.ball_coord.numpy()[0])
    ball_dof = int(m.ball_dof.numpy()[0])
    for env in range(num_envs):
        expected = Rotation.from_rotvec(dofs_np[env, ball_dof : ball_dof + 3]).as_quat()
        np.testing.assert_allclose(coords.numpy()[env, ball_coord : ball_coord + 4], expected, atol=1e-6)

    out = wp.zeros((num_envs, n_dofs), dtype=wp.float32, device="cpu")
    gather_joint_coordinates(m, coords, out)
    np.testing.assert_allclose(out.numpy(), dofs_np, atol=1e-5)


def test_gather_is_invariant_to_quaternion_sign() -> None:
    """``q`` and ``-q`` are the same rotation and must decode to the same rotation vector."""
    m = _map()
    n_dofs, n_coords = sum(DOF_COUNTS), sum(COORD_COUNTS)
    c = int(m.ball_coord.numpy()[0])
    base = np.zeros((1, n_coords), dtype=np.float32)
    base[0, c : c + 4] = Rotation.from_rotvec([0.2, -0.5, 0.1]).as_quat()

    decoded = []
    for sign in (1.0, -1.0):
        coords = base.copy()
        coords[0, c : c + 4] *= sign
        out = wp.zeros((1, n_dofs), dtype=wp.float32, device="cpu")
        gather_joint_coordinates(m, wp.array(coords, dtype=wp.float32, device="cpu"), out)
        decoded.append(out.numpy().copy())
    np.testing.assert_allclose(decoded[0], decoded[1], atol=1e-6)


def test_scatter_only_touches_masked_environments() -> None:
    """Resets are staggered, so an unmasked environment's coordinates must not move."""
    m = _map()
    n_dofs, n_coords = sum(DOF_COUNTS), sum(COORD_COUNTS)
    coords_np = np.full((2, n_coords), 0.25, dtype=np.float32)
    coords = wp.array(coords_np, dtype=wp.float32, device="cpu")
    dofs = wp.array(np.full((2, n_dofs), 0.3, dtype=np.float32), dtype=wp.float32, device="cpu")

    scatter_joint_coordinates(m, dofs, coords, wp.array(np.array([True, False]), dtype=wp.bool, device="cpu"))
    assert not np.allclose(coords.numpy()[0], coords_np[0])
    np.testing.assert_array_equal(coords.numpy()[1], coords_np[1])
