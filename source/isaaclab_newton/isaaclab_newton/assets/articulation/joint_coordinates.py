# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion between Newton's joint coordinate space and Isaac Lab's DOF space.

Newton stores a ball joint as a 4-component unit quaternion against 3 DOFs, so an articulation
containing one has more joint coordinates than DOFs. Free and distance joints have the same kind
of mismatch, but Isaac Lab's view excludes free joints and this module rejects distance joints, so
ball is the only mismatched layout this map converts. Every other joint type has one coordinate
per DOF, so the two spaces coincide for most assets and :func:`build_joint_coordinate_tables`
reports ``required = False`` for them.

Isaac Lab addresses joints by DOF index throughout -- ``joint_names``, ``find_joints``,
``SceneEntityCfg.joint_ids`` -- so joint positions have to be exposed in DOF space to stay
consistent with ``joint_vel``, ``default_joint_pos`` and the joint gains, all of which already are.

The rotation vector is the representation consistent with ``joint_qd``, which holds angular velocity
in the joint frame. PhysX also reports spherical-joint positions as an axis-angle vector projected
onto the DOF axes; sign and frame parity with PhysX has not been verified numerically.
"""

from __future__ import annotations

from typing import NamedTuple

import warp as wp

_BALL_LAYOUT = (4, 3)
"""``(coordinates, DOFs)`` of a ball joint -- the only mismatched layout this map converts; free
joints are excluded from Lab's view, and distance joints (7 against 6) are rejected."""


@wp.kernel(enable_backward=False)
def gather_single_coord_dofs(
    coords: wp.array2d(dtype=wp.float32),
    dof_index: wp.array(dtype=wp.int32),
    coord_index: wp.array(dtype=wp.int32),
    dofs: wp.array2d(dtype=wp.float32),
):
    """Copy joints whose coordinate count equals their DOF count."""
    env, i = wp.tid()
    dofs[env, dof_index[i]] = coords[env, coord_index[i]]


@wp.kernel(enable_backward=False)
def gather_ball_dofs(
    coords: wp.array2d(dtype=wp.float32),
    dof_index: wp.array(dtype=wp.int32),
    coord_index: wp.array(dtype=wp.int32),
    dofs: wp.array2d(dtype=wp.float32),
):
    """Quaternion -> rotation vector for each ball joint.

    ``wp.quat_to_axis_angle`` resolves the double cover itself -- it flips the axis with the sign of
    ``w`` and returns an angle in ``[0, pi]`` -- so a fixed pose always decodes to the same vector.
    """
    env, i = wp.tid()
    c = coord_index[i]
    d = dof_index[i]
    axis, angle = wp.quat_to_axis_angle(
        wp.quat(coords[env, c + 0], coords[env, c + 1], coords[env, c + 2], coords[env, c + 3])
    )
    rotvec = axis * angle
    dofs[env, d + 0] = rotvec[0]
    dofs[env, d + 1] = rotvec[1]
    dofs[env, d + 2] = rotvec[2]


@wp.kernel(enable_backward=False)
def scatter_single_coord_dofs(
    dofs: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    dof_index: wp.array(dtype=wp.int32),
    coord_index: wp.array(dtype=wp.int32),
    coords: wp.array2d(dtype=wp.float32),
):
    """Inverse of :func:`gather_single_coord_dofs`, for the selected environments."""
    env, i = wp.tid()
    if env_mask[env]:
        coords[env, coord_index[i]] = dofs[env, dof_index[i]]


@wp.kernel(enable_backward=False)
def scatter_ball_dofs(
    dofs: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    dof_index: wp.array(dtype=wp.int32),
    coord_index: wp.array(dtype=wp.int32),
    coords: wp.array2d(dtype=wp.float32),
):
    """Rotation vector -> quaternion for each ball joint, for the selected environments."""
    env, i = wp.tid()
    if not env_mask[env]:
        return
    c = coord_index[i]
    d = dof_index[i]
    rotvec = wp.vec3(dofs[env, d + 0], dofs[env, d + 1], dofs[env, d + 2])
    angle = wp.length(rotvec)
    q = wp.quat_identity()
    if angle > 1.0e-9:
        q = wp.quat_from_axis_angle(rotvec / angle, angle)
    coords[env, c + 0] = q[0]
    coords[env, c + 1] = q[1]
    coords[env, c + 2] = q[2]
    coords[env, c + 3] = q[3]


class JointCoordinateTables(NamedTuple):
    """Index tables mapping an articulation view's joint coordinates to its DOFs and back.

    A plain data record, not an object with behavior -- :func:`build_joint_coordinate_tables`
    produces one, and :func:`gather_joint_coordinates` / :func:`scatter_joint_coordinates` consume
    one. Fields are only meaningful when ``required`` is True; the ``as_wp`` conversion in
    :func:`build_joint_coordinate_tables` is skipped otherwise, so on a required=False table the
    remaining fields are placeholder empty arrays.
    """

    required: bool
    single_dof: wp.array
    single_coord: wp.array
    ball_dof: wp.array
    ball_coord: wp.array


def build_joint_coordinate_tables(coord_counts: list[int], dof_counts: list[int], device) -> JointCoordinateTables:
    """Build the index tables mapping an articulation view's joint coordinates to its DOFs and back.

    Built from the view's own per-joint counts, which are in the column order of
    :meth:`~newton.ArticulationView.get_dof_positions` and already exclude the free root joint,
    fixed joints and loop-closing joints.

    Args:
        coord_counts: Coordinates per selected joint (``ArticulationView.joint_coord_counts``).
        dof_counts: DOFs per selected joint (``ArticulationView.joint_dof_counts``).
        device: Device to allocate the index tables on.

    Returns:
        The index tables, with ``required`` set when the articulation has a ball joint.

    Raises:
        NotImplementedError: If a joint's coordinate and DOF counts differ in any way other than the
            ball-joint layout, which would otherwise be silently decoded as a quaternion.
    """
    single_dof: list[int] = []
    single_coord: list[int] = []
    ball_dof: list[int] = []
    ball_coord: list[int] = []

    coord, dof = 0, 0
    for n_coords, n_dofs in zip(coord_counts, dof_counts, strict=True):
        if n_coords == n_dofs:
            single_dof.extend(range(dof, dof + n_dofs))
            single_coord.extend(range(coord, coord + n_coords))
        elif (n_coords, n_dofs) == _BALL_LAYOUT:
            ball_dof.append(dof)
            ball_coord.append(coord)
        else:
            raise NotImplementedError(
                f"Joint with {n_coords} coordinates against {n_dofs} DOFs has no coordinate"
                " conversion; only ball joints (4 against 3) are supported."
            )
        coord += n_coords
        dof += n_dofs

    required = bool(ball_dof)
    if not required:
        empty = wp.array([], dtype=wp.int32, device=device)
        return JointCoordinateTables(False, empty, empty, empty, empty)
    as_wp = lambda values: wp.array(values, dtype=wp.int32, device=device)  # noqa: E731
    return JointCoordinateTables(True, as_wp(single_dof), as_wp(single_coord), as_wp(ball_dof), as_wp(ball_coord))


def gather_joint_coordinates(tables: JointCoordinateTables, coords: wp.array, dofs: wp.array) -> None:
    """Write the DOF-space view of ``coords`` into ``dofs``.

    Args:
        tables: Index tables from :func:`build_joint_coordinate_tables`.
        coords: Newton's joint coordinate array for this view.
        dofs: DOF-space destination [rad or m, depending on joint type].
    """
    num_envs = coords.shape[0]
    for kernel, dof_index, coord_index in (
        (gather_single_coord_dofs, tables.single_dof, tables.single_coord),
        (gather_ball_dofs, tables.ball_dof, tables.ball_coord),
    ):
        wp.launch(
            kernel,
            dim=(num_envs, dof_index.shape[0]),
            inputs=[coords, dof_index, coord_index, dofs],
            device=coords.device,
        )


def scatter_joint_coordinates(
    tables: JointCoordinateTables, dofs: wp.array, coords: wp.array, env_mask: wp.array
) -> None:
    """Write ``dofs`` back into ``coords`` for the selected environments.

    Scoping this to the written environments matters: resets are staggered, and ``dofs`` for
    the other environments only holds the last post-step gather. An all-environment scatter
    would overwrite their live ``joint_q`` with an exp(log(q)) round trip of itself, perturbing
    state that was never written.

    Args:
        tables: Index tables from :func:`build_joint_coordinate_tables`.
        dofs: DOF-space joint positions [rad or m, depending on joint type].
        coords: Newton's joint coordinate array to write into.
        env_mask: Per-environment boolean selection of the environments that were written.
    """
    for kernel, dof_index, coord_index in (
        (scatter_single_coord_dofs, tables.single_dof, tables.single_coord),
        (scatter_ball_dofs, tables.ball_dof, tables.ball_coord),
    ):
        wp.launch(
            kernel,
            dim=(env_mask.shape[0], dof_index.shape[0]),
            inputs=[dofs, env_mask, dof_index, coord_index, coords],
            device=coords.device,
        )
