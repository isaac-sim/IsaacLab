# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless tests for the Newton joint coordinate/DOF conversion."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation.joint_coordinates import (
    JointCoordinateTables,
    build_joint_coordinate_tables,
    gather_joint_coordinates,
    scatter_joint_coordinates,
)

# One revolute, one ball, one revolute -- the layout that hides an off-by-one when the tables are
# built by walking the model instead of the view's own per-joint counts.
COORD_COUNTS = [1, 4, 1]
DOF_COUNTS = [1, 3, 1]

# Two balls, straddling a revolute on each side -- the second ball's offsets (ball_coord[1] = 5,
# ball_dof[1] = 4) are not exercised by the single-ball layout above.
TWO_BALL_COORD_COUNTS = [1, 4, 4, 1]
TWO_BALL_DOF_COUNTS = [1, 3, 3, 1]


def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Reference exp map, independent of the kernel under test. Returns ``(x, y, z, w)``."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rotvec / angle
    return np.concatenate([axis * np.sin(0.5 * angle), [np.cos(0.5 * angle)]])


def _map() -> JointCoordinateTables:
    return build_joint_coordinate_tables(COORD_COUNTS, DOF_COUNTS, "cpu")


def test_tables_cover_every_dof() -> None:
    """Every DOF of every joint must be tabulated exactly once."""
    m = _map()
    covered = list(m.single_dof.numpy()) + [b + k for b in m.ball_dof.numpy() for k in range(3)]
    assert sorted(covered) == list(range(sum(DOF_COUNTS)))
    assert sorted(list(m.single_coord.numpy()) + [b + k for b in m.ball_coord.numpy() for k in range(4)]) == list(
        range(sum(COORD_COUNTS))
    )


def test_two_ball_tables_cover_every_dof() -> None:
    """A second ball joint's offsets are not a simple repeat of the first's."""
    m = build_joint_coordinate_tables(TWO_BALL_COORD_COUNTS, TWO_BALL_DOF_COUNTS, "cpu")
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


def test_joint_pos_is_dof_shaped_on_a_ball_jointed_mock() -> None:
    """Regression for the original bug: :attr:`ArticulationData.joint_pos` must be DOF-shaped
    even when the underlying view's ``joint_q`` is wider (a ball joint). This exercises
    ``_create_simulation_bindings`` end to end, unlike the coordinate-table-only tests
    above -- it fails on the pre-fix binding, which read ``get_dof_positions()`` (coordinate
    space) straight into ``joint_pos`` with no conversion, so ``joint_pos.shape[1]`` would be
    ``sum(TWO_BALL_COORD_COUNTS) == 10`` instead of ``sum(TWO_BALL_DOF_COUNTS) == 8``.
    """
    import isaaclab_newton.assets.articulation.articulation_data as newton_data_module
    from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
    from isaaclab_newton.test.fixtures.views import MockNewtonArticulationView

    num_instances = 2
    num_dofs = sum(TWO_BALL_DOF_COUNTS)
    num_bodies = len(TWO_BALL_DOF_COUNTS) + 1

    mock_view = MockNewtonArticulationView(
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=num_dofs,
        device="cpu",
        is_fixed_base=True,
        joint_coord_counts=TWO_BALL_COORD_COUNTS,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    mock_model = MagicMock()
    mock_model.world_count = num_instances
    mock_model.gravity = wp.array(
        np.tile(np.array([[0.0, 0.0, -9.81]], dtype=np.float32), (num_instances + 1, 1)),
        dtype=wp.vec3f,
        device="cpu",
    )
    mock_model.articulation_count = num_instances
    mock_model.max_joints_per_articulation = num_bodies
    mock_model.max_dofs_per_articulation = num_dofs
    mock_model.joint_dof_count = num_instances * num_dofs
    mock_model.body_count = num_instances * num_bodies

    mock_manager = MagicMock()
    mock_manager.get_model.return_value = mock_model
    mock_manager.get_state_0.return_value = MagicMock()
    mock_manager.get_state_1.return_value = MagicMock()
    mock_manager.get_control.return_value = MagicMock()

    original_sim_manager = newton_data_module.SimulationManager
    newton_data_module.SimulationManager = mock_manager
    try:
        data = ArticulationData(mock_view, "cpu")
    finally:
        newton_data_module.SimulationManager = original_sim_manager

    assert data.joint_pos.torch.shape[1] == num_dofs


def test_map_is_inert_without_ball_joints() -> None:
    """An articulation whose joints all have one coordinate per DOF needs no conversion."""
    assert not build_joint_coordinate_tables([1, 1, 1], [1, 1, 1], "cpu").required
    assert not build_joint_coordinate_tables([], [], "cpu").required


def test_unsupported_layout_is_rejected() -> None:
    """A distance joint (7 coordinates, 6 DOFs) must not be decoded as a quaternion."""
    with pytest.raises(NotImplementedError, match="7 coordinates against 6 DOFs"):
        build_joint_coordinate_tables([7], [6], "cpu")


@pytest.mark.parametrize("num_envs", [1, 2])
def test_scatter_then_gather_round_trips(num_envs: int) -> None:
    """DOF values survive a trip through coordinate space, at one environment and at two."""
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
        expected = _rotvec_to_quat(dofs_np[env, ball_dof : ball_dof + 3])
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
    base[0, c : c + 4] = _rotvec_to_quat(np.array([0.2, -0.5, 0.1]))

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
