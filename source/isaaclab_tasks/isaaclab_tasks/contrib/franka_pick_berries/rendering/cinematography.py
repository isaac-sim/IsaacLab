# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shot-by-shot camera for demonstration videos of the berry sorting sequence.

Each phase of :class:`~..control.sorting.BerrySortSequence` selects a shot: an establishing view of the room, macro
views of the grasps, tracking views of the carries and close-ups of the releases, ending with an orbit of the glass
bowl. Every shot moves slowly, and the camera blends from one shot to the next. Time is simulation time, so that a
recording plays at the same pace however slowly it renders.
"""

import math

import numpy as np

from ..scene.tableware import BOWL, REJECT

# A camera looking from this side of the workcell shows the room behind it, as the default workcell view does.
ROOM_AZIMUTH = -47.0
"""Azimuth [deg] of the camera about its subject from which the room fills the background."""

CLOSE_SHOTS = ("macro", "grasp", "release")
"""Shots that use the viewer's depth of field, if any."""


def _smooth(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _around(subject: np.ndarray, distance: float, azimuth: float, elevation: float) -> np.ndarray:
    """Return a camera position ``distance`` [m] from ``subject`` at ``azimuth`` and ``elevation`` [deg]."""
    a, e = math.radians(azimuth), math.radians(elevation)
    return subject + distance * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


class SortCinematography:
    """Choose and move the camera for each phase of the sorting sequence."""

    def __init__(self, transition: float = 1.2):
        """Configure the blend duration [s] between consecutive shots."""
        self.transition = transition
        self.reset()

    def reset(self) -> None:
        """Start again from the establishing shot."""
        self.shot = None
        self.shot_time = 0.0
        self.pose = None
        self.start_pose = None

    def update(self, viewer, sequence, berries: dict, tcp: np.ndarray, dt: float) -> None:
        """Move ``viewer``'s camera for the current phase of ``sequence``.

        Args:
            viewer: The berry viewer, whose camera this places.
            sequence: The sorting sequence, after its command for this step.
            berries: The berries, in sorting order.
            tcp: Gripper position [m].
            dt: Simulation time since the last update [s].
        """
        index = max(sequence.index, 0)
        berry = list(berries.values())[index]
        viewer.follow(berry)
        shot = self._choose(sequence, index)
        if shot != self.shot:
            self.shot, self.shot_time, self.start_pose = shot, 0.0, self.pose
        else:
            self.shot_time += dt
        center = berry.positions().mean(0) + berry.offset
        eye, target, fov = getattr(self, f"_{shot}")(index, center, np.asarray(tcp, float), self.shot_time)
        if self.start_pose is not None:
            # Blend from where the previous shot left the camera.
            blend = _smooth(self.shot_time / self.transition)
            start_eye, start_target, start_fov = self.start_pose
            eye = start_eye + blend * (eye - start_eye)
            target = start_target + blend * (target - start_target)
            fov = start_fov + blend * (fov - start_fov)
        self.pose = (eye, target, fov)
        # Close shots may blur their background; wide shots keep the room sharp.
        viewer.place_camera(
            eye, target, fov, focus=float(np.linalg.norm(target - eye)), depth_of_field=shot in CLOSE_SHOTS
        )

    @staticmethod
    def _choose(sequence, index: int) -> str:
        phase = sequence.phase
        if phase == "Opening":
            return "establishing"
        if phase == "Sequence complete":
            return "finale"
        if "Crush" in phase or "Gentle grasp" in phase:
            # The third grasp is shown wider, which keeps the pace up.
            return "grasp" if index == 2 else "macro"
        if "Carry" in phase or "grasp lost" in phase:
            return "carry"
        if "Release" in phase:
            return "release"
        return "approach"

    # Each shot returns the camera position and target [m] and the vertical field of view [deg].

    @staticmethod
    def _establishing(index, center, tcp, time):
        """Push in from the room toward the punnet."""
        u = _smooth(time / 3.0)
        subject = np.array([0.46, 0.05, 0.03])
        return _around(subject, 1.1 - 0.5 * u, ROOM_AZIMUTH + 8.0 * (1.0 - u), 24.0 + 10.0 * u), subject, 48.0 - 8 * u

    @staticmethod
    def _approach(index, center, tcp, time):
        """Three-quarter view of the berry, drifting while the gripper comes down."""
        subject = center + np.array([0.0, 0.0, 0.01])
        azimuth = (-70.0, -30.0, -60.0)[index] + 4.0 * time
        return _around(subject, 0.22, azimuth, (35.0, 35.0, 60.0)[index]), subject, 34.0

    @staticmethod
    def _macro(index, center, tcp, time):
        """Close, across the fingers and just over the punnet rim, orbiting slowly: the berry deforms between them."""
        subject = center + np.array([0.0, 0.0, 0.003])
        azimuth = (-20.0, 25.0, 0.0)[index] + 5.0 * time
        return _around(subject, 0.09, azimuth, 26.0), subject, 24.0

    @staticmethod
    def _grasp(index, center, tcp, time):
        """Medium view of a quicker grasp, from the room side, clear of the arm."""
        subject = center + np.array([0.0, 0.0, 0.005])
        return _around(subject, 0.16, ROOM_AZIMUTH + 4.0 * time, 35.0), subject, 30.0

    @staticmethod
    def _carry(index, center, tcp, time):
        """Track the gripper and its berry, with the room behind."""
        subject = 0.5 * (tcp + center)
        return _around(subject, 0.34, ROOM_AZIMUTH - 10.0 + 3.0 * time, 24.0), subject, 40.0

    @staticmethod
    def _release(index, center, tcp, time):
        """Push in on the reject dish or the glass bowl as the berry lands."""
        x, y = (REJECT if index == 0 else BOWL)[:2]
        subject = np.array([x, y, 0.015])
        distance = 0.19 - 0.04 * _smooth(time / 3.0)
        return _around(subject, distance, ROOM_AZIMUTH + (10.0 if index == 0 else -15.0), 32.0), subject, 30.0

    @staticmethod
    def _finale(index, center, tcp, time):
        """Orbit the glass bowl and its two berries, then pull back to the room."""
        subject = np.array([BOWL[0], BOWL[1], 0.015])
        out = _smooth((time - 3.5) / 2.5)
        distance = 0.2 + 0.7 * out
        elevation = 30.0 - 6.0 * out
        target = subject + out * (np.array([0.46, 0.05, 0.03]) - subject)
        # The orbit ends on the room side, so that the pull-back closes on the room.
        azimuth = ROOM_AZIMUTH - 60.0 * (1.0 - _smooth(time / 6.0))
        return _around(target, distance, azimuth, elevation), target, 30.0 + 16.0 * out
