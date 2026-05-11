# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cable-tangent-tracking hand orientation utilities.

Computes dynamic IK rotation targets so the finger claw gap (finger Z-axis)
straddles the cable. The cable tangent aligns with the finger X-axis,
allowing the 15mm claw gap to constrain the cable regardless of cable angle.

All quaternions are in ``(x, y, z, w)`` order (xyzw, scalar last).

Public API:
    BASE_HAND_DOWN_QUAT  -- base hand-down quaternion (xyzw)
    compute_cable_tangent(cable_positions, index)
    compute_hand_quat_for_cable(cable_tangent, base_quat=None)

Usage in RL env::

    from cable_orientation_utils import (
        BASE_HAND_DOWN_QUAT,
        compute_cable_tangent,
        compute_hand_quat_for_cable,
    )

    # Caller supplies arm/EE offsets from its own SSOT, e.g. test_newton_clip_routing.
    tangent = compute_cable_tangent(cable_pos, nearest_index)
    rot = compute_hand_quat_for_cable(tangent)  # xyzw quaternion
"""

import math

import numpy as np


def _quat_multiply(q1, q2):
    """Multiply two quaternions in ``(x, y, z, w)`` order (xyzw, scalar last).

    Internal helper. Not part of the public API.

    Args:
        q1: First quaternion ``[x1, y1, z1, w1]``.
        q2: Second quaternion ``[x2, y2, z2, w2]``.

    Returns:
        Product quaternion ``[x, y, z, w]`` as ``np.float32`` of shape ``[4]``.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def compute_cable_tangent(cable_positions, index):
    """Compute cable tangent direction at a given body index.

    Uses central difference for interior bodies, one-sided at endpoints.

    Args:
        cable_positions: [N_bodies, 3] numpy array of cable body positions.
        index: Body index at which to compute tangent.

    Returns:
        Unit tangent vector [3]. Falls back to (0, 1, 0) if degenerate.
    """
    n = len(cable_positions)
    if n < 2:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    if index <= 0:
        tangent = cable_positions[1] - cable_positions[0]
    elif index >= n - 1:
        tangent = cable_positions[-1] - cable_positions[-2]
    else:
        tangent = cable_positions[index + 1] - cable_positions[index - 1]

    norm = np.linalg.norm(tangent)
    if norm < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (tangent / norm).astype(np.float32)


# Base quaternion: hand down, fingers aligned for cable along Y-axis.
# 180° rotation around axis (cos(pi/8), sin(pi/8), 0) in world frame.
# Stored as quaternion in (x, y, z, w) order (xyzw, scalar last):
#   q = [n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2), cos(theta/2)]
# For theta=180°: q = [n_x, n_y, n_z, 0] = [cos(pi/8), sin(pi/8), 0, 0].
_COS_PI8 = math.cos(math.pi / 8)
_SIN_PI8 = math.sin(math.pi / 8)
BASE_HAND_DOWN_QUAT = np.array([_COS_PI8, _SIN_PI8, 0.0, 0.0], dtype=np.float32)

# Reference cable direction for the base quaternion.
_REF_CABLE_DIR = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def compute_hand_quat_for_cable(cable_tangent, base_quat=None):
    """Compute hand orientation quaternion that tracks cable direction.

    Rotates from the base orientation (cable along Y) so that the finger
    X-axis aligns with the actual cable tangent. The hand stays as close to
    "pointing down" as the cable angle permits.

    Args:
        cable_tangent: ``[3]`` unit vector of cable direction at grasp point.
        base_quat: ``[4]`` base quaternion in ``(x, y, z, w)`` order for the
            cable-along-Y reference. Defaults to :data:`BASE_HAND_DOWN_QUAT`.

    Returns:
        ``[4]`` ``np.float32`` quaternion in ``(x, y, z, w)`` order.
    """
    if base_quat is None:
        base_quat = BASE_HAND_DOWN_QUAT

    cable_tangent = np.asarray(cable_tangent, dtype=np.float32)
    t_norm = np.linalg.norm(cable_tangent)
    if t_norm < 1e-8:
        return base_quat.copy()
    cable_tangent = cable_tangent / t_norm

    cos_a = float(np.dot(_REF_CABLE_DIR, cable_tangent))
    cos_a = max(-1.0, min(1.0, cos_a))

    # Nearly aligned with reference — no rotation needed
    if cos_a > 0.9999:
        return base_quat.copy()

    # Anti-parallel — 180° around world Z; (x, y, z, w) = [0, 0, 1, 0].
    if cos_a < -0.9999:
        q_delta = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return _quat_multiply(q_delta, base_quat)

    # General case: axis-angle rotation from Y to cable_tangent
    axis = np.cross(_REF_CABLE_DIR, cable_tangent)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return base_quat.copy()
    axis = axis / axis_norm

    half_angle = math.acos(cos_a) / 2.0
    s = math.sin(half_angle)
    # axis-angle to quaternion in (x, y, z, w) order.
    q_delta = np.array(
        [
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
            math.cos(half_angle),
        ],
        dtype=np.float32,
    )

    return _quat_multiply(q_delta, base_quat)


def _compute_hand_quats_batch(
    cable_bodies_list, body_q, bws, ee_body_offset, ee_to_fingertip, franka_num_joints, base_quat=None
):
    """Compute per-world hand rotation quaternions for both arms.

    Internal helper. Not part of the public API. Callers in this code base use
    :func:`compute_cable_tangent` + :func:`compute_hand_quat_for_cable` directly
    and supply their own EE offsets from the env-side SSOT.

    For each world, finds the nearest cable body to each arm's fingertip
    and computes the cable-tracking quaternion.

    Args:
        cable_bodies_list: List of lists — cable body global indices per world.
        body_q: Full ``body_q`` ``np.ndarray`` from the physics state.
        bws: ``body_world_start`` ``np.ndarray``.
        ee_body_offset: EE body index offset within arm (caller-supplied).
        ee_to_fingertip: Distance from EE link to fingertip ``[m]``.
        franka_num_joints: Number of joints per arm (body offset for right arm).
        base_quat: Base hand-down quaternion ``[4]`` in ``(x, y, z, w)`` order.
            Defaults to :data:`BASE_HAND_DOWN_QUAT`.

    Returns:
        Tuple of ``(left_quats, right_quats)``, each ``[N, 4]`` ``np.float32``
        array of per-world quaternions in ``(x, y, z, w)`` order.
    """
    if base_quat is None:
        base_quat = BASE_HAND_DOWN_QUAT

    N = len(cable_bodies_list)
    quats_left = np.zeros((N, 4), dtype=np.float32)
    quats_right = np.zeros((N, 4), dtype=np.float32)

    for w in range(N):
        ws = bws[w]
        cable_pos = body_q[cable_bodies_list[w], :3]

        # Left arm (body offset = 0)
        ee_left = body_q[ws + ee_body_offset][:3]
        tip_left = ee_left.copy()
        tip_left[2] -= ee_to_fingertip
        dists_l = np.linalg.norm(cable_pos - tip_left, axis=1)
        nearest_l = int(np.argmin(dists_l))
        tangent_l = compute_cable_tangent(cable_pos, nearest_l)
        quats_left[w] = compute_hand_quat_for_cable(tangent_l, base_quat)

        # Right arm (body offset = franka_num_joints)
        ee_right = body_q[ws + franka_num_joints + ee_body_offset][:3]
        tip_right = ee_right.copy()
        tip_right[2] -= ee_to_fingertip
        dists_r = np.linalg.norm(cable_pos - tip_right, axis=1)
        nearest_r = int(np.argmin(dists_r))
        tangent_r = compute_cable_tangent(cable_pos, nearest_r)
        quats_right[w] = compute_hand_quat_for_cable(tangent_r, base_quat)

    return quats_left, quats_right
