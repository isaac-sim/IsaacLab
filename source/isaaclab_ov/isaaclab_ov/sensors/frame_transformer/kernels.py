# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for the OVPhysX frame transformer sensor."""

import warp as wp

# ---- Frame transformer update kernel ----


# ---- Gather body pose kernel ----


@wp.kernel
def gather_body_pose_kernel(
    env_mask: wp.array(dtype=wp.bool),
    pose_buffer: wp.array(dtype=wp.transformf),
    dst_flat_indices: wp.array(dtype=wp.int32),
    raw_transforms: wp.array(dtype=wp.transformf),
):
    """Copy a single body's per-env pose into the flat raw transforms buffer.

    For each env in the launch, copies ``pose_buffer[env]`` into
    ``raw_transforms[dst_flat_indices[env]]``. Skips envs whose ``env_mask`` is False.

    The pose buffer is a ``(num_envs,)`` ``wp.transformf`` array filled in place
    from a single ``RIGID_BODY_POSE`` view via ``OvPhysxView.read_into``. One
    launch per tracked body fills the body's slot column in the flat
    ``raw_transforms`` buffer.

    Args:
        env_mask: Active environment mask, shape ``(num_envs,)``.
        pose_buffer: Per-env world pose [m, dimensionless], shape ``(num_envs,)``,
            dtype ``wp.transformf`` in ``(px, py, pz, qx, qy, qz, qw)`` format.
        dst_flat_indices: Destination slot in ``raw_transforms`` per env, shape ``(num_envs,)``.
        raw_transforms: Destination flat pose buffer [m, dimensionless], shape
            ``(num_envs * num_unique_bodies,)``, dtype ``wp.transformf``.
    """
    env_id = wp.tid()
    if not env_mask[env_id]:
        return
    raw_transforms[dst_flat_indices[env_id]] = pose_buffer[env_id]
