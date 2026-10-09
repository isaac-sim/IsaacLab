# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera choreography for the Rizon--Sharpa teapot pouring example."""

from __future__ import annotations

import numpy as np


class PourCameraTracker:
    """Suppress camera jitter with a critically damped 2.5 Hz natural frequency.

    One tracker belongs to one playback. Its exact time integration follows the
    linearly interpolated camera commands between timestamps, independent of the
    physics step rate. Rewinding time resets the filter. The low-frequency tracking
    delay is approximately 0.13 seconds; shot distance remains unchanged.
    """

    def __init__(self) -> None:
        """Initialize an empty camera history."""
        self._time_s: float | None = None
        self._command: np.ndarray | None = None
        self._view: np.ndarray | None = None
        self._velocity = np.zeros((2, 3), dtype=np.float64)

    def update(
        self,
        time_s: float,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Filter the camera commands while preserving their eye--target distance.

        Args:
            time_s: Current playback time [s].
            eye: Unfiltered camera eye position in the world frame [m].
            target: Unfiltered camera target position in the world frame [m].

        Returns:
            Smoothed camera eye and target positions in the world frame [m].
        """
        command = np.asarray((eye, target), dtype=np.float64)
        if self._time_s is None or time_s < self._time_s:
            self._view = command.copy()
            self._velocity.fill(0.0)
        elif time_s > self._time_s:
            dt = time_s - self._time_s
            omega = 2.0 * np.pi * 2.5
            slope = (command - self._command) / dt
            # Exact critically damped response to a linearly changing command.
            error = self._view - self._command + 2.0 * slope / omega
            relative_velocity = self._velocity - slope
            transient = relative_velocity + omega * error
            decay = np.exp(-omega * dt)
            self._view = command - 2.0 * slope / omega + (error + transient * dt) * decay
            self._velocity = slope + (relative_velocity - omega * transient * dt) * decay
        self._time_s = time_s
        self._command = command
        view = self._view.copy()
        direction = view[0] - view[1]
        distance = np.linalg.norm(direction)
        if distance > 0.0:
            view[0] = view[1] + direction * (np.linalg.norm(command[0] - command[1]) / distance)
        return tuple(float(value) for value in view[0]), tuple(float(value) for value in view[1])


def pickup_video_view(
    time_s: float,
    container_height_w: float,
    *,
    container_base_height_w: float,
    manipulation_offset_x: float,
    recovery_start_time: float,
    recovery_time: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Follow pickup and pouring, then widen the view during recovery.

    Args:
        time_s: Simulation time [s].
        container_height_w: Planned teapot root height [m] in the world frame.
        container_base_height_w: Planned root height [m] before pouring.
        manipulation_offset_x: Translation of the manipulation along world X [m].
        recovery_start_time: Start of the return to an upright pot [s].
        recovery_time: Duration of that return [s].

    Returns:
        Camera eye and target positions [m] in the world frame.
    """
    approach = max(0.0, min(1.0, (time_s - 0.4) / 1.2))
    retreat = max(0.0, min(1.0, (time_s - recovery_start_time) / recovery_time))
    blend = approach**3 * (10.0 + approach * (-15.0 + 6.0 * approach))
    blend *= 1.0 - retreat**3 * (10.0 + retreat * (-15.0 + 6.0 * retreat))
    height = 0.5 * (container_height_w - container_base_height_w)
    camera_distance = max(0.0, 2.0 * height)
    eye = tuple(
        (1.0 - blend) * a + blend * b
        for a, b in zip(
            (0.40 + manipulation_offset_x, -1.6, 2.05 + 0.5 * height),
            (0.04 + manipulation_offset_x, -0.48 - 1.2 * camera_distance, 1.44 + height),
            strict=True,
        )
    )
    target = tuple(
        (1.0 - blend) * a + blend * b
        for a, b in zip(
            (-0.30 + manipulation_offset_x, 0.0, 1.35 + 0.5 * height),
            (-0.12 + manipulation_offset_x, 0.0, 1.065 + height),
            strict=True,
        )
    )
    return eye, target


def pour_closeup_view(
    time_s: float,
    container_pose_w: tuple[tuple[float, float, float], tuple[float, float, float, float]],
    *,
    spout_local_pos: tuple[float, float, float],
    bowl_top_pos_w: tuple[float, float, float],
    robot_base_pos_w: tuple[float, float, float],
    rise_start_time: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], bool]:
    """Frame the grasp, outgoing water, and bowl impact using measured object poses.

    Camera moves use quintic blends with zero speed and acceleration at their endpoints.
    The overview's eye--target distance is fixed at 1.55 m, including during the upward pour.
    A two-second surface-rendering interval sits inside the established stream close-up;
    particles remain visible throughout the other shots. This changes presentation only.

    Args:
        time_s: Simulation time [s].
        container_pose_w: Measured teapot root position [m] and XYZW quaternion in the world frame.
        spout_local_pos: Outlet center [m] in the teapot root frame.
        bowl_top_pos_w: Center of the bowl's upper opening [m] in the world frame.
        robot_base_pos_w: Robot root position [m] in the world frame.
        rise_start_time: Start of the upward pouring motion [s].

    Returns:
        Camera eye and target positions [m] in the world frame, and whether to render the
        reconstructed water surface instead of particles at this time.
    """
    pot_position = np.asarray(container_pose_w[0], dtype=np.float64)
    pot_quaternion = np.asarray(container_pose_w[1], dtype=np.float64)
    outlet_local = np.asarray(spout_local_pos, dtype=np.float64)
    bowl_top = np.asarray(bowl_top_pos_w, dtype=np.float64)
    robot_base = np.asarray(robot_base_pos_w, dtype=np.float64)
    quaternion_vector = pot_quaternion[:3]
    outlet_rotated = outlet_local + 2.0 * np.cross(
        quaternion_vector, np.cross(quaternion_vector, outlet_local) + pot_quaternion[3] * outlet_local
    )
    outlet = pot_position + outlet_rotated

    overview_target = np.array(
        (
            0.45 * robot_base[0] + 0.55 * bowl_top[0],
            bowl_top[1],
            max(robot_base[2] + 0.12, bowl_top[2] + 0.28) + 0.10 * max(0.0, outlet[2] - bowl_top[2] - 0.15),
        )
    )
    overview_eye = overview_target + (0.60, -1.30, 0.60)
    grasp_target = pot_position + (-0.05, 0.0, 0.06)
    grasp_eye = grasp_target + (0.12, -0.43, 0.24)
    stream_target = outlet + (0.0, 0.0, -0.032)
    stream_eye = stream_target + (0.14, -0.29, 0.16)
    bowl_target = bowl_top + (0.0, 0.0, -0.012)
    bowl_eye = bowl_target + (0.12, -0.22, 0.17)

    if time_s < 2.0:
        source_eye, source_target = overview_eye, overview_target
        destination_eye, destination_target = grasp_eye, grasp_target
        progress = (time_s - 0.4) / 1.6
    elif time_s < rise_start_time + 0.2:
        source_eye, source_target = grasp_eye, grasp_target
        destination_eye, destination_target = stream_eye, stream_target
        progress = (time_s - rise_start_time + 1.2) / 1.4
    elif time_s < rise_start_time + 3.8:
        source_eye, source_target = stream_eye, stream_target
        destination_eye, destination_target = bowl_eye, bowl_target
        progress = (time_s - rise_start_time - 2.7) / 1.1
    else:
        source_eye, source_target = bowl_eye, bowl_target
        destination_eye, destination_target = overview_eye, overview_target
        progress = (time_s - rise_start_time - 4.8) / 1.5

    progress = max(0.0, min(1.0, progress))
    blend = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
    eye = (1.0 - blend) * source_eye + blend * destination_eye
    target = (1.0 - blend) * source_target + blend * destination_target
    show_surface = rise_start_time + 0.4 <= time_s < rise_start_time + 2.4
    return tuple(float(value) for value in eye), tuple(float(value) for value in target), show_surface
