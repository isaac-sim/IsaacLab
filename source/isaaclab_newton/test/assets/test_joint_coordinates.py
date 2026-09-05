# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless tests for the Newton joint coordinate/DOF conversion."""

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation.joint_coordinates import JointCoordinateMap

# One revolute, one ball, one revolute -- the layout that hides an off-by-one when the tables are
# built by walking the model instead of the view's own per-joint counts.
COORD_COUNTS = [1, 4, 1]
DOF_COUNTS = [1, 3, 1]


def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Reference exp map, independent of the kernel under test. Returns ``(x, y, z, w)``."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rotvec / angle
    return np.concatenate([axis * np.sin(0.5 * angle), [np.cos(0.5 * angle)]])


def _map() -> JointCoordinateMap:
    return JointCoordinateMap(COORD_COUNTS, DOF_COUNTS, "cpu")


def test_tables_cover_every_dof() -> None:
    """Every DOF of every joint must be tabulated exactly once."""
    m = _map()
    covered = list(m.single_dof.numpy()) + [b + k for b in m.ball_dof.numpy() for k in range(3)]
    assert sorted(covered) == list(range(sum(DOF_COUNTS)))
    assert sorted(list(m.single_coord.numpy()) + [b + k for b in m.ball_coord.numpy() for k in range(4)]) == list(
        range(sum(COORD_COUNTS))
    )


def test_map_is_inert_without_ball_joints() -> None:
    """An articulation whose joints all have one coordinate per DOF needs no conversion."""
    assert not JointCoordinateMap([1, 1, 1], [1, 1, 1], "cpu").required
    assert not JointCoordinateMap([], [], "cpu").required


def test_unsupported_layout_is_rejected() -> None:
    """A distance joint (7 coordinates, 6 DOFs) must not be decoded as a quaternion."""
    with pytest.raises(NotImplementedError, match="7 coordinates against 6 DOFs"):
        JointCoordinateMap([7], [6], "cpu")


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

    m.scatter(dofs, coords, mask)
    # The quaternion the scatter wrote must match an independent exp map.
    ball_coord = int(m.ball_coord.numpy()[0])
    ball_dof = int(m.ball_dof.numpy()[0])
    for env in range(num_envs):
        expected = _rotvec_to_quat(dofs_np[env, ball_dof : ball_dof + 3])
        np.testing.assert_allclose(coords.numpy()[env, ball_coord : ball_coord + 4], expected, atol=1e-6)

    out = wp.zeros((num_envs, n_dofs), dtype=wp.float32, device="cpu")
    m.gather(coords, out)
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
        m.gather(wp.array(coords, dtype=wp.float32, device="cpu"), out)
        decoded.append(out.numpy().copy())
    np.testing.assert_allclose(decoded[0], decoded[1], atol=1e-6)


def test_scatter_only_touches_masked_environments() -> None:
    """Resets are staggered, so an unmasked environment's coordinates must not move."""
    m = _map()
    n_dofs, n_coords = sum(DOF_COUNTS), sum(COORD_COUNTS)
    coords_np = np.full((2, n_coords), 0.25, dtype=np.float32)
    coords = wp.array(coords_np, dtype=wp.float32, device="cpu")
    dofs = wp.array(np.full((2, n_dofs), 0.3, dtype=np.float32), dtype=wp.float32, device="cpu")

    m.scatter(dofs, coords, wp.array(np.array([True, False]), dtype=wp.bool, device="cpu"))
    assert not np.allclose(coords.numpy()[0], coords_np[0])
    np.testing.assert_array_equal(coords.numpy()[1], coords_np[1])
