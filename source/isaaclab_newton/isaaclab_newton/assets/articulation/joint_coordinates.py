# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conversion between Newton's joint coordinate space and IsaacLab's DOF space.

Newton stores a ball joint as a 4-component unit quaternion against 3 DOFs, so an articulation
containing one has more joint coordinates (``Model.joint_q_start``) than DOFs
(``Model.joint_qd_start``). Every other joint type has one coordinate per DOF, so the two spaces
coincide for most assets and :class:`JointCoordinateMap` reports ``required = False`` for them.

IsaacLab addresses joints by DOF index throughout -- ``joint_names``, ``find_joints``,
``SceneEntityCfg.joint_ids`` -- so joint positions have to be exposed in DOF space to stay
consistent with ``joint_vel``, ``default_joint_pos`` and the joint gains, all of which already are.

A quaternion double-covers SO(3): ``q`` and ``-q`` are the same rotation. The gather forces
``w >= 0`` before taking the log map so that a fixed pose always decodes to the same rotation
vector; without it the sign can flip between steps and inject discontinuities into observations.
"""

from __future__ import annotations

from typing import Any

import warp as wp
from newton import JointType


def as_warp_indices(selection):
    """Normalise an environment selection to a warp array.

    ``Articulation._resolve_env_ids`` may return either a warp array or a torch tensor.
    """
    if isinstance(selection, wp.array):
        return selection
    return wp.from_torch(selection.contiguous())


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
    """Quaternion -> rotation vector for each ball joint."""
    env, i = wp.tid()
    c = coord_index[i]
    d = dof_index[i]
    axis = wp.vec3(coords[env, c + 0], coords[env, c + 1], coords[env, c + 2])
    w = coords[env, c + 3]
    if w < 0.0:  # same rotation, continuous branch
        axis = -axis
        w = -w
    length = wp.length(axis)
    scale = float(2.0)  # small-angle limit of angle / |axis|
    if length > 1.0e-8:
        scale = 2.0 * wp.atan2(length, wp.clamp(w, -1.0, 1.0)) / length
    dofs[env, d + 0] = axis[0] * scale
    dofs[env, d + 1] = axis[1] * scale
    dofs[env, d + 2] = axis[2] * scale


@wp.func
def _scatter_ball(
    dofs: wp.array2d(dtype=wp.float32),
    env: wp.int32,
    i: wp.int32,
    dof_index: wp.array(dtype=wp.int32),
    coord_index: wp.array(dtype=wp.int32),
    coords: wp.array2d(dtype=wp.float32),
):
    """Rotation vector -> quaternion for one ball joint of one environment."""
    c = coord_index[i]
    d = dof_index[i]
    axis = wp.vec3(dofs[env, d + 0], dofs[env, d + 1], dofs[env, d + 2])
    angle = wp.length(axis)
    if angle < 1.0e-9:
        coords[env, c + 0] = 0.0
        coords[env, c + 1] = 0.0
        coords[env, c + 2] = 0.0
        coords[env, c + 3] = 1.0
    else:
        s = wp.sin(0.5 * angle) / angle
        coords[env, c + 0] = axis[0] * s
        coords[env, c + 1] = axis[1] * s
        coords[env, c + 2] = axis[2] * s
        coords[env, c + 3] = wp.cos(0.5 * angle)


@wp.kernel(enable_backward=False)
def scatter_single_coord_dofs(
    dofs: wp.array2d(dtype=wp.float32),
    env_index: wp.array(dtype=Any),
    dof_index: wp.array(dtype=wp.int32),
    coord_index: wp.array(dtype=wp.int32),
    coords: wp.array2d(dtype=wp.float32),
):
    """Inverse of :func:`gather_single_coord_dofs`, for the selected environments."""
    e, i = wp.tid()
    env = wp.int32(env_index[e])
    coords[env, coord_index[i]] = dofs[env, dof_index[i]]


@wp.kernel(enable_backward=False)
def scatter_single_coord_dofs_masked(
    dofs: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    dof_index: wp.array(dtype=wp.int32),
    coord_index: wp.array(dtype=wp.int32),
    coords: wp.array2d(dtype=wp.float32),
):
    """Mask-selected counterpart, launched over every environment so it stays graph-capturable."""
    env, i = wp.tid()
    if env_mask[env]:
        coords[env, coord_index[i]] = dofs[env, dof_index[i]]


@wp.kernel(enable_backward=False)
def scatter_ball_dofs(
    dofs: wp.array2d(dtype=wp.float32),
    env_index: wp.array(dtype=Any),
    dof_index: wp.array(dtype=wp.int32),
    coord_index: wp.array(dtype=wp.int32),
    coords: wp.array2d(dtype=wp.float32),
):
    """Rotation vector -> quaternion for each ball joint, for the selected environments."""
    e, i = wp.tid()
    env = wp.int32(env_index[e])
    _scatter_ball(dofs, env, i, dof_index, coord_index, coords)


@wp.kernel(enable_backward=False)
def scatter_ball_dofs_masked(
    dofs: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
    dof_index: wp.array(dtype=wp.int32),
    coord_index: wp.array(dtype=wp.int32),
    coords: wp.array2d(dtype=wp.float32),
):
    """Mask-selected counterpart, launched over every environment so it stays graph-capturable."""
    env, i = wp.tid()
    if env_mask[env]:
        _scatter_ball(dofs, env, i, dof_index, coord_index, coords)


class JointCoordinateMap:
    """Per-articulation index tables mapping joint coordinates to DOFs and back.

    Offsets are local to the articulation's joint-coordinate slice -- the array
    ``ArticulationView.get_dof_positions`` returns, which excludes the free root joint -- so they
    are accumulated from zero rather than read off the model's global ``joint_q_start``. The walk
    starts at the view's own first joint, since a heterogeneous scene may register another
    articulation ahead of this one.
    """

    @classmethod
    def inert(cls) -> JointCoordinateMap:
        """A map that converts nothing, for articulations that carry no joints."""
        obj = cls.__new__(cls)
        obj.required = False
        return obj

    def __init__(self, model, num_dofs: int, first_joint: int, device):
        single_dof: list[int] = []
        single_coord: list[int] = []
        ball_dof: list[int] = []
        ball_coord: list[int] = []

        joint_type = model.joint_type.numpy()
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        coord, dof = 0, 0
        for j in range(first_joint, len(joint_type) - 1):
            n_coords = int(q_start[j + 1]) - int(q_start[j])
            n_dofs = int(qd_start[j + 1]) - int(qd_start[j])
            if n_dofs == 0 or int(joint_type[j]) == int(JointType.FREE):
                continue  # fixed joints hold no state; the root is not part of the slice
            if n_coords == n_dofs:
                for k in range(n_dofs):
                    single_dof.append(dof + k)
                    single_coord.append(coord + k)
            else:
                ball_dof.append(dof)
                ball_coord.append(coord)
            coord += n_coords
            dof += n_dofs
            if dof >= num_dofs:
                break  # any joints past this belong to loop closures, not to the DOF view

        self.required = len(ball_dof) > 0
        if not self.required:
            return
        as_wp = lambda values: wp.array(values, dtype=wp.int32, device=device)  # noqa: E731
        self.single_dof = as_wp(single_dof)
        self.single_coord = as_wp(single_coord)
        self.ball_dof = as_wp(ball_dof)
        self.ball_coord = as_wp(ball_coord)

    def gather(self, coords: wp.array, dofs: wp.array) -> None:
        """Write the DOF-space view of ``coords`` into ``dofs``."""
        num_envs = coords.shape[0]
        wp.launch(
            gather_single_coord_dofs,
            dim=(num_envs, self.single_dof.shape[0]),
            inputs=[coords, self.single_dof, self.single_coord, dofs],
            device=coords.device,
        )
        wp.launch(
            gather_ball_dofs,
            dim=(num_envs, self.ball_dof.shape[0]),
            inputs=[coords, self.ball_dof, self.ball_coord, dofs],
            device=coords.device,
        )

    def scatter(
        self,
        dofs: wp.array,
        coords: wp.array,
        env_index: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write ``dofs`` back into ``coords`` for the given environments.

        Restricting this to the written environments matters: resets are staggered, so an
        all-environment scatter would send every quaternion through a log-map round trip on
        almost every step and inject float32 noise into the loop-closure constraints.

        Args:
            dofs: DOF-space joint positions [rad or m, depending on joint type].
            coords: Newton's joint coordinate array to write into.
            env_index: Environment indices that were written. Mutually exclusive with ``env_mask``.
            env_mask: Per-environment boolean selection. Mutually exclusive with ``env_index``.
                Taken by the mask write paths, which are documented as graph-capturable, so it is
                consumed directly rather than compacted to indices on the host.
        """
        if env_mask is not None:
            for kernel, dof_index, coord_index in (
                (scatter_single_coord_dofs_masked, self.single_dof, self.single_coord),
                (scatter_ball_dofs_masked, self.ball_dof, self.ball_coord),
            ):
                wp.launch(
                    kernel,
                    dim=(env_mask.shape[0], dof_index.shape[0]),
                    inputs=[dofs, env_mask, dof_index, coord_index, coords],
                    device=coords.device,
                )
            return
        for kernel, dof_index, coord_index in (
            (scatter_single_coord_dofs, self.single_dof, self.single_coord),
            (scatter_ball_dofs, self.ball_dof, self.ball_coord),
        ):
            wp.launch(
                kernel,
                dim=(env_index.shape[0], dof_index.shape[0]),
                inputs=[dofs, env_index, dof_index, coord_index, coords],
                device=coords.device,
            )
