# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion between Newton's joint coordinate space and Isaac Lab's DOF space.

Newton stores a ball joint as a 4-component unit quaternion against 3 DOFs, so an articulation
containing one has more joint coordinates than DOFs. Every other joint type Isaac Lab exposes has
one coordinate per DOF, so the two spaces coincide for most assets and :class:`JointCoordinateMap`
reports ``required = False`` for them.

Isaac Lab addresses joints by DOF index throughout -- ``joint_names``, ``find_joints``,
``SceneEntityCfg.joint_ids`` -- so joint positions have to be exposed in DOF space to stay
consistent with ``joint_vel``, ``default_joint_pos`` and the joint gains, all of which already are.

The rotation vector is the representation consistent with ``joint_qd``, which holds angular velocity
in the joint frame, and with PhysX, whose spherical-joint positions are the axis-angle vector
projected onto the DOF axes.
"""

from __future__ import annotations

import warp as wp

_BALL_LAYOUT = (4, 3)
"""``(coordinates, DOFs)`` of a ball joint -- the only pair Newton exposes where the two differ."""


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


class JointCoordinateMap:
    """Index tables mapping an articulation view's joint coordinates to its DOFs and back.

    Built from the view's own per-joint counts, which are in the column order of
    :meth:`~newton.ArticulationView.get_dof_positions` and already exclude the free root joint,
    fixed joints and loop-closing joints.

    Args:
        coord_counts: Coordinates per selected joint (``ArticulationView.joint_coord_counts``).
        dof_counts: DOFs per selected joint (``ArticulationView.joint_dof_counts``).
        device: Device to allocate the index tables on.

    Raises:
        NotImplementedError: If a joint's coordinate and DOF counts differ in any way other than the
            ball-joint layout, which would otherwise be silently decoded as a quaternion.
    """

    def __init__(self, coord_counts: list[int], dof_counts: list[int], device):
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

        self.required = bool(ball_dof)
        if not self.required:
            return
        as_wp = lambda values: wp.array(values, dtype=wp.int32, device=device)  # noqa: E731
        self.single_dof = as_wp(single_dof)
        self.single_coord = as_wp(single_coord)
        self.ball_dof = as_wp(ball_dof)
        self.ball_coord = as_wp(ball_coord)

    def gather(self, coords: wp.array, dofs: wp.array) -> None:
        """Write the DOF-space view of ``coords`` into ``dofs``.

        Args:
            coords: Newton's joint coordinate array for this view.
            dofs: DOF-space destination [rad or m, depending on joint type].
        """
        num_envs = coords.shape[0]
        for kernel, dof_index, coord_index in (
            (gather_single_coord_dofs, self.single_dof, self.single_coord),
            (gather_ball_dofs, self.ball_dof, self.ball_coord),
        ):
            wp.launch(
                kernel,
                dim=(num_envs, dof_index.shape[0]),
                inputs=[coords, dof_index, coord_index, dofs],
                device=coords.device,
            )

    def scatter(self, dofs: wp.array, coords: wp.array, env_mask: wp.array) -> None:
        """Write ``dofs`` back into ``coords`` for the selected environments.

        Scoping this to the written environments matters: resets are staggered, so an
        all-environment scatter would send every quaternion through a log-map round trip on almost
        every step and add float32 noise to the loop-closure constraints.

        Args:
            dofs: DOF-space joint positions [rad or m, depending on joint type].
            coords: Newton's joint coordinate array to write into.
            env_mask: Per-environment boolean selection of the environments that were written.
        """
        for kernel, dof_index, coord_index in (
            (scatter_single_coord_dofs, self.single_dof, self.single_coord),
            (scatter_ball_dofs, self.ball_dof, self.ball_coord),
        ):
            wp.launch(
                kernel,
                dim=(env_mask.shape[0], dof_index.shape[0]),
                inputs=[dofs, env_mask, dof_index, coord_index, coords],
                device=coords.device,
            )
