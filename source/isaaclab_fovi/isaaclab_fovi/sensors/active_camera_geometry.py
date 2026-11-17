# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared active-camera stereo geometry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_CameraPose = tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]


@dataclass(frozen=True)
class PanTiltMountGeometry:
    """Neutral root and camera-local poses for an image-aligned pan/tilt mount."""

    base_pose: _CameraPose
    camera_local_pose: _CameraPose


def stereo_baseline_fixation_axis(
    left_quat_xyzw: tuple[float, float, float, float],
    right_in_left: tuple[float, ...] | None,
    *,
    manual_stereo_horizontal_separation_m: float | None = None,
    horizontal_axis_parent: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> tuple[float, float]:
    """Project the stereo baseline into the mount's fixation controls.

    Canonical fixation commands are ``(tilt, pan)`` around the parent-frame
    +X and +Z axes. The returned unit direction is the command whose neutral
    optical-axis motion points along the left-to-right baseline projected into
    the optical tangent plane.
    """
    from scipy.spatial.transform import Rotation as R

    left_quat = np.asarray(left_quat_xyzw, dtype=np.float64)
    if left_quat.shape != (4,) or not np.isfinite(left_quat).all():
        raise ValueError("left_quat_xyzw must contain four finite values.")
    left_rot = R.from_quat(left_quat)

    if manual_stereo_horizontal_separation_m is not None:
        separation = float(manual_stereo_horizontal_separation_m)
        if not np.isfinite(separation) or separation <= 0.0:
            raise ValueError(
                "manual_stereo_horizontal_separation_m must be finite and > 0, "
                f"got {manual_stereo_horizontal_separation_m!r}."
            )
        axis_parent = np.asarray(horizontal_axis_parent, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis_parent))
        if not np.isfinite(axis_parent).all() or axis_norm <= 1.0e-8:
            raise ValueError("horizontal_axis_parent must be finite and non-zero.")
        baseline_parent = separation * axis_parent / axis_norm
    else:
        if right_in_left is None:
            raise ValueError("right_in_left is required to derive the calibrated vergence axis.")
        transform = np.asarray(right_in_left, dtype=np.float64)
        if transform.size != 16 or not np.isfinite(transform).all():
            raise ValueError("right_in_left must contain 16 finite transform values.")
        baseline_left = transform.reshape(4, 4)[:3, 3]
        baseline_parent = left_rot.apply(baseline_left)

    optical_axis_parent = left_rot.apply(np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    baseline_tangent = baseline_parent - (np.dot(baseline_parent, optical_axis_parent) * optical_axis_parent)
    if float(np.linalg.norm(baseline_tangent)) <= 1.0e-8:
        raise ValueError("Stereo baseline has no resolvable component in the fixation tangent plane.")

    control_jacobian = np.column_stack(
        (
            np.cross((1.0, 0.0, 0.0), optical_axis_parent),
            np.cross((0.0, 0.0, 1.0), optical_axis_parent),
        )
    )
    control_axis, _, rank, _ = np.linalg.lstsq(
        control_jacobian,
        baseline_tangent,
        rcond=None,
    )
    if rank != 2:
        raise ValueError("Camera optical axis is singular for the configured tilt/pan mount controls.")
    control_norm = float(np.linalg.norm(control_axis))
    if control_norm <= 1.0e-8:
        raise ValueError("Stereo baseline produces no resolvable tilt/pan command.")
    unit = control_axis / control_norm
    return float(unit[0]), float(unit[1])


def optical_axis_offset(
    camera_quat_xyzw: tuple[float, float, float, float],
    offset_m: float,
) -> tuple[float, float, float]:
    """Return a parent-frame offset along the ROS-camera optical axis."""
    offset_m = float(offset_m)
    if offset_m == 0.0:
        return (0.0, 0.0, 0.0)

    from scipy.spatial.transform import Rotation as R

    forward = R.from_quat(np.asarray(camera_quat_xyzw, dtype=np.float64)).apply(
        np.asarray((0.0, 0.0, offset_m), dtype=np.float64)
    )
    return tuple(float(v) for v in forward)


def centered_stereo_aperture_poses(
    center_pos: tuple[float, float, float],
    left_quat_xyzw: tuple[float, float, float, float],
    right_in_left: tuple[float, ...] | None,
    *,
    use_stereo: bool,
    manual_stereo_horizontal_separation_m: float | None = None,
    horizontal_axis_parent: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float, float]],
    tuple[tuple[float, float, float], tuple[float, float, float, float]] | None,
]:
    """Return centered left/right aperture poses in the parent frame.

    ``center_pos`` is the midpoint between the two camera aperture centers.
    ``right_in_left`` is the calibrated right-camera transform expressed in
    the left-camera aperture frame. ``manual_stereo_horizontal_separation_m``
    overrides that calibrated transform with a centered yoked baseline along
    ``horizontal_axis_parent``.
    """
    from scipy.spatial.transform import Rotation as R

    center = np.asarray(center_pos, dtype=np.float64)
    left_quat = np.asarray(left_quat_xyzw, dtype=np.float64)
    left_rot = R.from_quat(left_quat)

    if not use_stereo:
        left_pose = (
            tuple(float(v) for v in center),
            tuple(float(v) for v in left_quat),
        )
        return left_pose, None
    if right_in_left is None:
        if manual_stereo_horizontal_separation_m is None:
            raise ValueError("right_in_left is required when use_stereo=True.")

    if manual_stereo_horizontal_separation_m is not None:
        separation = float(manual_stereo_horizontal_separation_m)
        if separation <= 0.0:
            raise ValueError(
                "manual_stereo_horizontal_separation_m must be > 0 when set, "
                f"got {manual_stereo_horizontal_separation_m!r}."
            )
        axis = np.asarray(horizontal_axis_parent, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-8:
            raise ValueError("horizontal_axis_parent must be non-zero when manual stereo separation is set.")
        baseline_parent = separation * (axis / axis_norm)
        left_pos = center - 0.5 * baseline_parent
        right_pos = center + 0.5 * baseline_parent
        left_pose = (
            tuple(float(v) for v in left_pos),
            tuple(float(v) for v in left_quat),
        )
        right_pose = (
            tuple(float(v) for v in right_pos),
            tuple(float(v) for v in left_quat),
        )
        return left_pose, right_pose

    right_in_left_tf = np.asarray(right_in_left, dtype=np.float64).reshape(4, 4)
    baseline_parent = left_rot.apply(right_in_left_tf[:3, 3])
    left_pos = center - 0.5 * baseline_parent

    left_tf = np.eye(4)
    left_tf[:3, :3] = left_rot.as_matrix()
    left_tf[:3, 3] = left_pos
    right_tf = left_tf @ right_in_left_tf
    right_rot = R.from_matrix(right_tf[:3, :3])

    left_pose = (
        tuple(float(v) for v in left_pos),
        tuple(float(v) for v in left_quat),
    )
    right_pose = (
        tuple(float(v) for v in right_tf[:3, 3]),
        tuple(float(v) for v in right_rot.as_quat()),
    )
    return left_pose, right_pose


def stereo_aperture_poses(
    left_pos: tuple[float, float, float],
    left_quat_xyzw: tuple[float, float, float, float],
    right_in_left: tuple[float, ...] | None,
    *,
    use_stereo: bool,
    manual_stereo_horizontal_separation_m: float | None = None,
    horizontal_axis_parent: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> tuple[_CameraPose, _CameraPose | None]:
    """Return the configured camera poses without recentering calibration.

    Calibrated camera configs define ``left_pos`` as the actual left aperture
    position and ``right_in_left`` relative to it. A manual yoked pair instead
    treats that position as its midpoint, matching the active-camera presets.
    """
    if manual_stereo_horizontal_separation_m is not None:
        return centered_stereo_aperture_poses(
            left_pos,
            left_quat_xyzw,
            right_in_left,
            use_stereo=use_stereo,
            manual_stereo_horizontal_separation_m=manual_stereo_horizontal_separation_m,
            horizontal_axis_parent=horizontal_axis_parent,
        )

    left_pose = (left_pos, left_quat_xyzw)
    if not use_stereo:
        return left_pose, None
    if right_in_left is None:
        raise ValueError("right_in_left is required when use_stereo=True.")

    from scipy.spatial.transform import Rotation as R

    left_transform = np.eye(4, dtype=np.float64)
    left_transform[:3, :3] = R.from_quat(np.asarray(left_quat_xyzw, dtype=np.float64)).as_matrix()
    left_transform[:3, 3] = np.asarray(left_pos, dtype=np.float64)
    relative_transform = np.asarray(right_in_left, dtype=np.float64)
    if relative_transform.size != 16 or not np.isfinite(relative_transform).all():
        raise ValueError("right_in_left must contain 16 finite transform values.")
    right_transform = left_transform @ relative_transform.reshape(4, 4)
    right_pose = (
        tuple(float(value) for value in right_transform[:3, 3]),
        tuple(float(value) for value in R.from_matrix(right_transform[:3, :3]).as_quat()),
    )
    return left_pose, right_pose


def pan_tilt_mount_geometry(
    camera_pos: tuple[float, float, float],
    camera_quat_xyzw: tuple[float, float, float, float],
    *,
    image_rotation_deg: float,
    camera_forward_offset_m: float,
    base_z_drop_m: float,
) -> PanTiltMountGeometry:
    """Factor a calibrated camera pose into an image-aligned pan/tilt mount.

    The mount's X and Z axes become policy-image right and up. Its Y axis is
    optical forward, matching the bundled pan/tilt asset. The returned local
    camera pose exactly reconstructs the calibrated pose at zero joint angles.
    """
    from scipy.spatial.transform import Rotation as R

    position = np.asarray(camera_pos, dtype=np.float64).copy()
    quaternion = np.asarray(camera_quat_xyzw, dtype=np.float64)
    rotation_deg = float(image_rotation_deg)
    forward_offset = float(camera_forward_offset_m)
    base_z_drop = float(base_z_drop_m)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("camera_pos must contain three finite values.")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("camera_quat_xyzw must contain four finite values.")
    if not math.isfinite(rotation_deg):
        raise ValueError("image_rotation_deg must be finite.")
    if not math.isfinite(forward_offset):
        raise ValueError("camera_forward_offset_m must be finite.")
    if not math.isfinite(base_z_drop) or base_z_drop < 0.0:
        raise ValueError("base_z_drop_m must be finite and non-negative.")

    camera_rotation = R.from_quat(quaternion)
    camera_matrix = camera_rotation.as_matrix()
    raw_right = camera_matrix[:, 0]
    raw_up = -camera_matrix[:, 1]
    optical_forward = camera_matrix[:, 2]
    theta = math.radians(rotation_deg)
    policy_right = math.cos(theta) * raw_right + math.sin(theta) * raw_up
    policy_up = -math.sin(theta) * raw_right + math.cos(theta) * raw_up
    mount_rotation = R.from_matrix(np.column_stack((policy_right, optical_forward, policy_up)))
    camera_local_rotation = mount_rotation.inv() * camera_rotation
    camera_local_position = mount_rotation.inv().apply(forward_offset * optical_forward)
    base_position = position
    base_position -= mount_rotation.apply((0.0, 0.0, base_z_drop))
    base_position -= forward_offset * optical_forward

    return PanTiltMountGeometry(
        base_pose=(
            tuple(float(value) for value in base_position),
            tuple(float(value) for value in mount_rotation.as_quat()),
        ),
        camera_local_pose=(
            tuple(float(value) for value in camera_local_position),
            tuple(float(value) for value in camera_local_rotation.as_quat()),
        ),
    )
