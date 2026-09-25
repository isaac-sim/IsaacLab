# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""KUKA-Allegro kinematics and production reset calibration for cube stacking."""

from __future__ import annotations

import torch

KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_LOWER: tuple[float, ...] = (
    -2.7,
    -1.9,
    -2.7,
    0.3,
    -2.7,
    -1.9,
    -3.05,
)
"""Lower joint boundary for the collision-safe wrist-diverse reset bank [rad]."""

KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_UPPER: tuple[float, ...] = (
    2.7,
    1.9,
    2.7,
    2.0694,
    2.7,
    1.9,
    3.05,
)
"""Upper joint boundary for the collision-safe wrist-diverse reset bank [rad]."""

KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES: tuple[str, ...] = tuple(
    f"{finger}_joint_{joint_id}" for finger in ("index", "middle", "ring", "thumb") for joint_id in range(4)
)
"""Canonical 16-joint Allegro hand order."""

KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES: tuple[tuple[str, ...], ...] = (
    tuple(f"index_joint_{joint_id}" for joint_id in range(4))
    + tuple(f"thumb_joint_{joint_id}" for joint_id in range(4)),
)
"""Joint names for the reset-authored index/thumb grasp."""

KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH = 0.08
"""Edge length of the KUKA-Allegro stacking cubes [m]."""

KUKA_ALLEGRO_LARGE_CUBE_RESTING_HEIGHT = 0.040
"""Center height for an 8 cm cube on the tabletop contact surface [m]."""

_FULL_HAND_INACTIVE_FINGER_POSE = (0.0, 0.9, 1.3, 1.3)
_FULL_HAND_OPEN_POSES = (
    (0.4395595789, 0.7841927409, 0.8926416039, 0.4153704941, 1.5507498789, 0.2781451344, 0.3000894189, 0.5697273612),
)
_FULL_HAND_CONTACT_POSES = (
    (0.3552525342, 0.7711680532, 0.9140521884, 0.4398036003, 1.5507498789, 0.2582621574, 0.2845720053, 0.6348888874),
)
_FULL_HAND_PRELOAD_POSES = (
    (0.3468218446, 0.7698655725, 0.9161932468, 0.4422469139, 1.5507498789, 0.2562738657, 0.2830202579, 0.6414050460),
)


def _full_hand_pair_command(active_pose: tuple[float, ...], pair_id: int) -> tuple[float, ...]:
    """Return one canonical all-hand command with two tucked fingers."""
    opposing_poses = [_FULL_HAND_INACTIVE_FINGER_POSE] * 3
    opposing_poses[pair_id] = active_pose[:4]
    return (*opposing_poses[0], *opposing_poses[1], *opposing_poses[2], *active_pose[4:])


KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_POSES: tuple[tuple[float, ...], ...] = _FULL_HAND_OPEN_POSES
"""Dynamically validated open posture for the reset grasp [rad]."""

KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_POSES: tuple[tuple[float, ...], ...] = _FULL_HAND_CONTACT_POSES
"""Dynamically validated zero-preload contact posture for the reset grasp [rad]."""

KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_COMMANDS: tuple[tuple[float, ...], ...] = tuple(
    _full_hand_pair_command(active_pose, pair_id) for pair_id, active_pose in enumerate(_FULL_HAND_OPEN_POSES)
)
"""Open all-hand command for the calibrated reset grasp [rad]."""

KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_COMMANDS: tuple[tuple[float, ...], ...] = tuple(
    _full_hand_pair_command(active_pose, pair_id) for pair_id, active_pose in enumerate(_FULL_HAND_CONTACT_POSES)
)
"""Contact all-hand command for the calibrated reset grasp [rad]."""

KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_PRELOAD_COMMANDS: tuple[tuple[float, ...], ...] = tuple(
    _full_hand_pair_command(active_pose, pair_id) for pair_id, active_pose in enumerate(_FULL_HAND_PRELOAD_POSES)
)
"""Preloaded all-hand target for reset-authored held cubes [rad]."""

KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.0669180581, -0.0325941718, 0.1014142311),
)
"""Cube-center offset for the calibrated index/thumb mid-face grasp [m]."""

KUKA_ALLEGRO_FULL_HAND_PALM_TO_HELD_CUBE_QUATERNIONS_XYZW: tuple[tuple[float, ...], ...] = (
    (0.0569160655, -0.7048124671, -0.0569160655, 0.7048124671),
)
"""Cube orientation for the calibrated reset grasp."""

# Retained for the reset bank's nominal palm rotation and anchor validation.
_KUKA_ALLEGRO_NOMINAL_PINCH_OFFSET = (0.0570965, -0.0375159, 0.0498749)

# The same nine XY anchors and five tool heights used by the Franka reset
# table, solved for a KUKA base rotated pi radians around world Z. Columns are
# near-grasp, pre-grasp, lifted/aligned, final-aligned, and release heights.
KUKA_ALLEGRO_STACK_ARM_POSES: tuple[tuple[tuple[float, ...], ...], ...] = (
    (
        (-0.3825908, -0.5482572, 0.1509951, 2.0693952, -0.1269447, -0.6562628, 2.2247175),
        (-0.4005797, -0.4509816, 0.1701806, 2.0693952, -0.1045209, -0.7474434, 2.2051455),
        (-0.4246691, -0.3052654, 0.1922202, 2.0693952, -0.0712347, -0.8593142, 2.1739231),
        (-0.4219077, -0.2316099, 0.1850011, 2.0693952, -0.0503769, -0.9044256, 2.1531426),
        (-0.4085717, -0.3594766, 0.1763572, 2.0693952, -0.0796055, -0.8205191, 2.1823202),
    ),
    (
        (-0.3448831, -0.5828326, 0.1411305, 2.0349160, -0.1528177, -0.5335495, 2.2612213),
        (-0.3645787, -0.4898472, 0.1641283, 2.0322451, -0.1310052, -0.6291900, 2.2428591),
        (-0.3894749, -0.3626934, 0.1911474, 2.0074525, -0.0960107, -0.7798138, 2.2140763),
        (-0.3854408, -0.3051329, 0.1846155, 1.9859250, -0.0730412, -0.8567360, 2.1949052),
        (-0.3726272, -0.4078041, 0.1725329, 2.0196409, -0.1031836, -0.7222126, 2.2197418),
    ),
    (
        (-0.3198590, -0.6544124, 0.1467614, 1.8878712, -0.1559575, -0.6102482, 2.2812354),
        (-0.3344194, -0.5711260, 0.1646751, 1.8853568, -0.1386560, -0.6960321, 2.2673653),
        (-0.3535277, -0.4574813, 0.1869835, 1.8619584, -0.1112775, -0.8321757, 2.2458918),
        (-0.3541568, -0.4061313, 0.1866436, 1.8415528, -0.0935478, -0.9023640, 2.2318901),
        (-0.3425062, -0.4979583, 0.1740330, 1.8734744, -0.1178729, -0.7800045, 2.2509278),
    ),
    (
        (-0.0585658, -0.5578322, 0.2214434, 2.0693952, -0.1804145, -0.6837048, 2.6726483),
        (-0.0846307, -0.4601540, 0.2483632, 2.0693952, -0.1468532, -0.7748841, 2.6426071),
        (-0.1212851, -0.3115247, 0.2812637, 2.0693952, -0.0984055, -0.8880310, 2.5950497),
        (-0.1180185, -0.2345408, 0.2711690, 2.0693952, -0.0681154, -0.9340362, 2.5632807),
        (-0.0975833, -0.3666447, 0.2581461, 2.0693952, -0.1107607, -0.8483889, 2.6083271),
    ),
    (
        (-0.0566120, -0.5760674, 0.1976127, 2.0582684, -0.2145674, -0.5262202, 2.6521982),
        (-0.0853326, -0.4820107, 0.2309950, 2.0555653, -0.1829249, -0.6229439, 2.6254473),
        (-0.1217197, -0.3525447, 0.2701851, 2.0304856, -0.1321284, -0.7747488, 2.5834598),
        (-0.1153702, -0.2927155, 0.2600127, 2.0087264, -0.0987421, -0.8518869, 2.5553878),
        (-0.0968959, -0.3979872, 0.2429271, 2.0428125, -0.1424548, -0.7163705, 2.5916495),
    ),
    (
        (-0.0631887, -0.6483918, 0.2007254, 1.9097990, -0.2136738, -0.6038489, 2.6305295),
        (-0.0837955, -0.5643822, 0.2259569, 1.9072659, -0.1893101, -0.6903568, 2.6109128),
        (-0.1109122, -0.4492543, 0.2573797, 1.8837011, -0.1507508, -0.8272914, 2.5805417),
        (-0.1117269, -0.3964783, 0.2566913, 1.8631622, -0.1257363, -0.8975502, 2.5606700),
        (-0.0952242, -0.4899906, 0.2390704, 1.8952972, -0.1600121, -0.7746181, 2.5876356),
    ),
    (
        (0.2974765, -0.5708679, 0.2104971, 2.0693952, -0.2240361, -0.5334828, 3.0293263),
        (0.2298164, -0.4793603, 0.2888694, 2.0693952, -0.2245162, -0.6299267, 3.0293263),
        (0.1798137, -0.3446982, 0.3419268, 2.0512142, -0.1633250, -0.7709101, 2.9768869),
        (0.2338689, -0.2783970, 0.2762795, 2.0292323, -0.1002546, -0.8465140, 2.9227638),
        (0.2536725, -0.3857591, 0.2583835, 2.0636742, -0.1481679, -0.7092287, 2.9628006),
    ),
    (
        (0.2226050, -0.6358752, 0.2476652, 1.9462043, -0.2645426, -0.5905172, 3.0008278),
        (0.1954592, -0.5503137, 0.2805830, 1.9436367, -0.2328024, -0.6785867, 2.9750432),
        (0.1596936, -0.4323842, 0.3214384, 1.9197652, -0.1825069, -0.8174679, 2.9350801),
        (0.1592096, -0.3773216, 0.3195958, 1.8989797, -0.1497299, -0.8882699, 2.9087442),
        (0.1806889, -0.4737504, 0.2972464, 1.9315089, -0.1944118, -0.7637973, 2.9442594),
    ),
    (
        (0.1915482, -0.7055001, 0.2475091, 1.8037012, -0.2606276, -0.6642364, 2.9447755),
        (0.1717512, -0.6277090, 0.2729192, 1.8012456, -0.2358132, -0.7444289, 2.9256492),
        (0.1452260, -0.5211652, 0.3056668, 1.7783656, -0.1969774, -0.8719424, 2.8962335),
        (0.1414101, -0.4721507, 0.3093123, 1.7583710, -0.1725857, -0.9376393, 2.8776603),
        (0.1590273, -0.5588414, 0.2885949, 1.7896326, -0.2073446, -0.8227323, 2.9039495),
    ),
)

_KUKA_JOINT_ORIGINS: tuple[tuple[tuple[float, float, float], tuple[float, float, float, float]], ...] = (
    ((0.0, 0.0, 0.1575), (1.0, 0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.1825), (-3.090862e-8, -3.090862e-8, 0.70710677, 0.70710677)),
    ((0.0, 0.184, 0.0), (-3.090862e-8, -3.090862e-8, 0.70710677, 0.70710677)),
    ((0.0, 0.0, 0.216), (0.70710677, 0.70710677, 0.0, 0.0)),
    ((0.0, 0.184, 0.0), (-3.090862e-8, 3.090862e-8, 0.70710677, 0.70710677)),
    ((0.0, 0.0607, 0.216), (0.70710677, 0.70710677, 0.0, 0.0)),
    ((0.0, 0.081, 0.0607), (-3.090862e-8, 3.090862e-8, 0.70710677, 0.70710677)),
)
_KUKA_LINK7_TO_MOUNT = (-2.1796957e-5, 1.0094006e-8, 0.0706024076)
_KUKA_MOUNT_TO_PALM_POSITION = (-0.008219, -0.02063, 0.08086)
_KUKA_MOUNT_TO_PALM_QUATERNION = (0.65328032, 0.2705985, -0.6532827, 0.27059752)


def _matrix_from_quaternion_wxyz(
    quaternion: tuple[float, float, float, float], reference: torch.Tensor
) -> torch.Tensor:
    """Return a rotation matrix for a normalized scalar-first quaternion."""
    w, x, y, z = quaternion
    return reference.new_tensor(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        )
    )


def matrix_from_quaternion_xyzw(
    quaternion: torch.Tensor | tuple[float, float, float, float],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return rotation matrices for normalized scalar-last quaternions."""
    quaternion = torch.as_tensor(quaternion, dtype=reference.dtype, device=reference.device)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-12)
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def quaternion_xyzw_from_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to normalized scalar-last quaternions."""
    if rotation.ndim < 2 or rotation.shape[-2:] != (3, 3):
        raise ValueError("Rotation matrices must have shape (..., 3, 3).")
    m00, m01, m02 = rotation[..., 0, 0], rotation[..., 0, 1], rotation[..., 0, 2]
    m10, m11, m12 = rotation[..., 1, 0], rotation[..., 1, 1], rotation[..., 1, 2]
    m20, m21, m22 = rotation[..., 2, 0], rotation[..., 2, 1], rotation[..., 2, 2]
    # The sign terms select the same hemisphere as the off-diagonal entries.
    # ``copysign`` is well defined at zero and avoids unstable divisions near
    # pi, which are common in the eight-bin wrist-yaw bank.
    x = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0))
    y = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0))
    z = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0))
    w = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    x = torch.copysign(x, m21 - m12)
    y = torch.copysign(y, m02 - m20)
    z = torch.copysign(z, m10 - m01)
    quaternion = torch.stack((x, y, z, w), dim=-1)
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-12)


def _rotation_vector_from_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Return shortest-path rotation vectors for rotation matrices."""
    quaternion = quaternion_xyzw_from_matrix(rotation)
    xyz, w = quaternion[..., :3], quaternion[..., 3:4]
    # q and -q encode the same rotation. Select non-negative w so the returned
    # vector always follows the shortest path and remains continuous at zero.
    xyz = torch.where(w < 0.0, -xyz, xyz)
    w = torch.abs(w)
    vector_norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, w.clamp_min(1.0e-12))
    scale = torch.where(vector_norm > 1.0e-7, angle / vector_norm, 2.0 * torch.ones_like(vector_norm))
    return xyz * scale


def kuka_allegro_palm_pose(joint_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the palm pose for the rotated KUKA-Allegro stack asset.

    The authored KUKA asset reaches toward negative X. The stack variant rotates
    its fixed base by pi around world Z. Apply that fixed transform to both the
    position and orientation.

    Args:
        joint_positions: KUKA arm positions [rad], shape ``(..., 7)``.

    Returns:
        Palm positions in environment-local coordinates [m], shape ``(..., 3)``,
        and rotation matrices, shape ``(..., 3, 3)``.
    """
    if joint_positions.ndim < 1 or joint_positions.shape[-1] != 7:
        raise ValueError("KUKA-Allegro reset forward kinematics expects seven joint positions.")
    batch_shape = joint_positions.shape[:-1]
    flat_joints = joint_positions.reshape(-1, 7)
    batch_size = flat_joints.shape[0]
    rotation = (
        torch.eye(3, dtype=joint_positions.dtype, device=joint_positions.device)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .clone()
    )
    position = torch.zeros((batch_size, 3), dtype=joint_positions.dtype, device=joint_positions.device)
    reference = flat_joints[0] if batch_size > 0 else joint_positions.new_zeros(7)
    for joint_id, (origin_position, origin_quaternion) in enumerate(_KUKA_JOINT_ORIGINS):
        origin = joint_positions.new_tensor(origin_position).expand(batch_size, -1)
        position += torch.bmm(rotation, origin.unsqueeze(-1)).squeeze(-1)
        rotation = torch.matmul(rotation, _matrix_from_quaternion_wxyz(origin_quaternion, reference))
        angle = flat_joints[:, joint_id]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        zeros = torch.zeros_like(angle)
        ones = torch.ones_like(angle)
        joint_rotation = torch.stack(
            (
                torch.stack((cosine, -sine, zeros), dim=1),
                torch.stack((sine, cosine, zeros), dim=1),
                torch.stack((zeros, zeros, ones), dim=1),
            ),
            dim=1,
        )
        rotation = torch.bmm(rotation, joint_rotation)

    for fixed_offset in (_KUKA_LINK7_TO_MOUNT, _KUKA_MOUNT_TO_PALM_POSITION):
        offset = joint_positions.new_tensor(fixed_offset).expand(batch_size, -1)
        position += torch.bmm(rotation, offset.unsqueeze(-1)).squeeze(-1)
    rotation = torch.matmul(
        rotation,
        _matrix_from_quaternion_wxyz(_KUKA_MOUNT_TO_PALM_QUATERNION, reference),
    )
    base_rotation = joint_positions.new_tensor(
        (
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    position = torch.matmul(position, base_rotation.T)
    rotation = torch.matmul(base_rotation, rotation)
    return (
        position.reshape(*batch_shape, 3),
        rotation.reshape(*batch_shape, 3, 3),
    )


def kuka_allegro_tool_pose(
    joint_positions: torch.Tensor,
    tool_offsets: torch.Tensor | tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a palm-fixed tool pose for the rotated KUKA-Allegro asset.

    Args:
        joint_positions: KUKA arm positions [rad], shape ``(..., 7)``.
        tool_offsets: Tool positions in the palm frame [m], shape ``(..., 3)``
            or one length-three tuple broadcast over all arm positions.

    Returns:
        Tool positions in environment-local coordinates [m], shape ``(..., 3)``,
        and palm rotation matrices, shape ``(..., 3, 3)``.
    """
    palm_position, palm_rotation = kuka_allegro_palm_pose(joint_positions)
    offset = torch.as_tensor(tool_offsets, dtype=joint_positions.dtype, device=joint_positions.device)
    offset = torch.broadcast_to(offset, joint_positions.shape[:-1] + (3,))
    tool_position = palm_position + torch.matmul(palm_rotation, offset.unsqueeze(-1)).squeeze(-1)
    return tool_position, palm_rotation


def kuka_allegro_grasp_pair_pose(
    joint_positions: torch.Tensor,
    pair_ids: torch.Tensor | int,
    tool_offsets_by_pair: tuple[tuple[float, float, float], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the calibrated tool pose selected by each grasp-pair ID.

    Args:
        joint_positions: KUKA arm positions [rad], shape ``(..., 7)``.
        pair_ids: Pair IDs in ``[0, 2]``, broadcastable to ``joint_positions``
            without its final joint dimension.
        tool_offsets_by_pair: Palm-frame grasp centers for each pair [m].

    Returns:
        Pair-center positions [m], shape ``(..., 3)``, and palm rotation
        matrices, shape ``(..., 3, 3)``.
    """
    pair_ids = torch.as_tensor(pair_ids, dtype=torch.long, device=joint_positions.device)
    pair_ids = torch.broadcast_to(pair_ids, joint_positions.shape[:-1])
    if bool(torch.any((pair_ids < 0) | (pair_ids >= len(tool_offsets_by_pair)))):
        raise ValueError(f"KUKA-Allegro grasp-pair IDs must be in [0, {len(tool_offsets_by_pair) - 1}].")
    offsets = joint_positions.new_tensor(tool_offsets_by_pair)[pair_ids]
    return kuka_allegro_tool_pose(joint_positions, offsets)


def kuka_allegro_pinch_pose(joint_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the nominal index/thumb pinch-center pose used by reset planning."""
    return kuka_allegro_tool_pose(joint_positions, _KUKA_ALLEGRO_NOMINAL_PINCH_OFFSET)


def solve_kuka_allegro_reset_ik(
    seed_joint_positions: torch.Tensor,
    target_positions: torch.Tensor,
    target_rotations: torch.Tensor,
    tool_offsets: torch.Tensor,
    *,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    max_iterations: int = 56,
    damping: float = 0.035,
    max_delta: float = 0.16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve batched collision-safe reset poses with damped least squares.

    The solver is intentionally independent of the simulation backend. It
    uses the reset FK above and a finite-difference geometric Jacobian, so the
    generated table has the same fixed pair center under both PhysX and
    Newton.

    Args:
        seed_joint_positions: Initial arm positions [rad], shape ``(N, 7)``.
        target_positions: Desired pair-center positions [m], shape ``(N, 3)``.
        target_rotations: Desired palm rotations, shape ``(N, 3, 3)``.
        tool_offsets: Pair-center offsets in the palm frame [m], shape
            ``(N, 3)``.
        joint_lower: Lower solver bounds [rad], shape ``(7,)``.
        joint_upper: Upper solver bounds [rad], shape ``(7,)``.
        max_iterations: Maximum DLS iterations.
        damping: DLS regularization.
        max_delta: Maximum joint update norm per iteration [rad].

    Returns:
        Solved positions [rad], position residuals [m], and orientation
        residuals [rad].
    """
    if seed_joint_positions.ndim != 2 or seed_joint_positions.shape[1] != 7:
        raise ValueError("Batched KUKA reset IK seeds must have shape (N, 7).")
    batch_size = seed_joint_positions.shape[0]
    if target_positions.shape != (batch_size, 3):
        raise ValueError("Batched KUKA reset IK positions must have shape (N, 3).")
    if target_rotations.shape != (batch_size, 3, 3):
        raise ValueError("Batched KUKA reset IK rotations must have shape (N, 3, 3).")
    if tool_offsets.shape != (batch_size, 3):
        raise ValueError("Batched KUKA reset IK tool offsets must have shape (N, 3).")
    if joint_lower.shape != (7,) or joint_upper.shape != (7,):
        raise ValueError("Batched KUKA reset IK joint bounds must have shape (7,).")
    if max_iterations < 1 or damping <= 0.0 or max_delta <= 0.0:
        raise ValueError("KUKA reset IK iteration, damping, and update limits must be positive.")

    # RSL-RL enables TF32 globally for policy throughput before constructing
    # the environment. The reset bank is built only once and its millimetre
    # validation must not depend on that training optimization. Re-enter with
    # IEEE FP32, then restore the caller's setting for PPO.
    if seed_joint_positions.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return solve_kuka_allegro_reset_ik(
                seed_joint_positions,
                target_positions,
                target_rotations,
                tool_offsets,
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                max_iterations=max_iterations,
                damping=damping,
                max_delta=max_delta,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = True

    joint_positions = seed_joint_positions.clone().clamp(min=joint_lower, max=joint_upper)
    identity = torch.eye(6, dtype=joint_positions.dtype, device=joint_positions.device).expand(batch_size, -1, -1)
    finite_difference = 1.0e-3
    for _ in range(max_iterations):
        position, rotation = kuka_allegro_tool_pose(joint_positions, tool_offsets)
        position_error = target_positions - position
        rotation_error = _rotation_vector_from_matrix(torch.matmul(target_rotations, rotation.transpose(-1, -2)))
        error = torch.cat((position_error, rotation_error), dim=1)

        perturbed = joint_positions[:, None, :].expand(-1, 7, -1).clone()
        diagonal = torch.arange(7, device=joint_positions.device)
        perturbed[:, diagonal, diagonal] += finite_difference
        perturbed_position, perturbed_rotation = kuka_allegro_tool_pose(
            perturbed.reshape(-1, 7),
            tool_offsets[:, None, :].expand(-1, 7, -1).reshape(-1, 3),
        )
        perturbed_position = perturbed_position.reshape(batch_size, 7, 3)
        perturbed_rotation = perturbed_rotation.reshape(batch_size, 7, 3, 3)
        linear_jacobian = ((perturbed_position - position[:, None, :]) / finite_difference).transpose(1, 2)
        relative_rotation = torch.matmul(perturbed_rotation, rotation[:, None].transpose(-1, -2))
        angular_jacobian = torch.stack(
            (
                relative_rotation[..., 2, 1] - relative_rotation[..., 1, 2],
                relative_rotation[..., 0, 2] - relative_rotation[..., 2, 0],
                relative_rotation[..., 1, 0] - relative_rotation[..., 0, 1],
            ),
            dim=2,
        ).transpose(1, 2) / (2.0 * finite_difference)
        jacobian = torch.cat((linear_jacobian, angular_jacobian), dim=1)
        normal = torch.bmm(jacobian, jacobian.transpose(1, 2)) + damping * damping * identity
        delta = torch.bmm(
            jacobian.transpose(1, 2),
            torch.linalg.solve(normal, error.unsqueeze(-1)),
        ).squeeze(-1)
        delta_norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        delta *= torch.clamp(max_delta / delta_norm.clamp_min(1.0e-12), max=1.0)
        joint_positions = (joint_positions + delta).clamp(min=joint_lower, max=joint_upper)

    position, rotation = kuka_allegro_tool_pose(joint_positions, tool_offsets)
    position_residual = torch.linalg.vector_norm(target_positions - position, dim=1)
    rotation_residual = torch.linalg.vector_norm(
        _rotation_vector_from_matrix(torch.matmul(target_rotations, rotation.transpose(-1, -2))),
        dim=1,
    )
    return joint_positions, position_residual, rotation_residual
