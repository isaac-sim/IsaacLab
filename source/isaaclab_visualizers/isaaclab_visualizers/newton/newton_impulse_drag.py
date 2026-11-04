# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Impulse Drag: aim pushes at bodies while the simulation is paused and apply them when it resumes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

IMPULSE_GAIN = 6.0
"""Speed given to a body per meter the arrow is dragged [m/s per m]."""
MAX_IMPULSE_SPEED = 6.0
"""Largest speed change an impulse can give a body [m/s]. Longer arrows are clamped to it."""
MAX_IMPULSES = 16
"""Impulses that can wait for the simulation to resume."""


@wp.kernel
def _apply_impulses(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    bodies: wp.array[int],
    points: wp.array[wp.vec3],
    impulses: wp.array[wp.vec3],
    inverse_dt: wp.array[float],
):
    i = wp.tid()
    body = bodies[i]
    if body < 0:
        return
    transform = body_q[body]
    point = wp.transform_point(transform, points[i])
    com = wp.transform_point(transform, body_com[body])
    force = impulses[i] * inverse_dt[0]
    wp.atomic_add(body_f, body, wp.spatial_vector(force, wp.cross(point - com, force)))
    bodies[i] = -1  # Applied once: the next solver substep finds the slot empty.


def impulse_color(fraction: float) -> tuple[float, float, float]:
    """Return the arrow color for an impulse of ``fraction`` of the largest one: green, yellow, then red."""
    t = min(max(float(fraction), 0.0), 1.0)
    if t < 0.5:
        return (2.0 * t, 1.0, 0.1)
    return (1.0, 2.0 * (1.0 - t), 0.1)


@dataclass
class Impulse:
    """One impulse waiting to be applied."""

    body: int
    """Index of the body the arrow starts on."""
    point: np.ndarray
    """Start of the arrow in the body's frame [m]."""
    vector: np.ndarray
    """Impulse [N·s], in the world frame."""
    fraction: float
    """Size of the impulse relative to the largest one, between 0 and 1."""


def make_impulse(body: int, point: np.ndarray, displacement: np.ndarray, mass: float) -> Impulse:
    """Turn an arrow dragged from a point on a body into the impulse it stands for.

    Args:
        body: Index of the body.
        point: Start of the arrow in the body's frame [m].
        displacement: The arrow, from the picked point to the cursor target, in the world frame [m].
        mass: Mass the impulse acts on [kg].

    Returns:
        The impulse. Its speed change is :data:`IMPULSE_GAIN` per meter of arrow, up to :data:`MAX_IMPULSE_SPEED`.
    """
    length = float(np.linalg.norm(displacement))
    if length < 1e-9:
        return Impulse(body, np.asarray(point, dtype=np.float32), np.zeros(3, dtype=np.float32), 0.0)
    speed = min(IMPULSE_GAIN * length, MAX_IMPULSE_SPEED)
    vector = (displacement / length * speed * mass).astype(np.float32)
    return Impulse(body, np.asarray(point, dtype=np.float32), vector, speed / MAX_IMPULSE_SPEED)


class ImpulseDrag:
    """Impulses aimed while paused, applied through the solver's force callback on the next step.

    The buffers have a fixed size and address so that the CUDA graph that records the solver substeps keeps working;
    the kernel empties a slot as it applies it.
    """

    def __init__(self, device, substep_dt: float):
        """Create the buffers.

        Args:
            device: Warp device of the model.
            substep_dt: Duration of one solver substep [s].
        """
        self._device = device
        self.entries: list[Impulse] = []
        self._bodies = wp.full(MAX_IMPULSES, -1, dtype=int, device=device)
        self._points = wp.zeros(MAX_IMPULSES, dtype=wp.vec3, device=device)
        self._impulses = wp.zeros(MAX_IMPULSES, dtype=wp.vec3, device=device)
        self._inverse_dt = wp.array([1.0 / substep_dt], dtype=float, device=device)

    def set_substep_dt(self, substep_dt: float) -> None:
        """Set the duration of one solver substep [s], which the impulse is applied over."""
        self._inverse_dt.assign(np.array([1.0 / substep_dt], dtype=np.float32))

    @property
    def full(self) -> bool:
        """Whether no more impulses can be added."""
        return len(self.entries) >= MAX_IMPULSES

    def add(self, impulse: Impulse) -> bool:
        """Queue an impulse, replacing the one already waiting on the same body.

        Returns:
            False when the impulse is empty or the queue is full, so nothing was queued.
        """
        if impulse.fraction <= 0.0:
            return False
        waiting = next((index for index, entry in enumerate(self.entries) if entry.body == impulse.body), None)
        if waiting is not None:
            self.entries[waiting] = impulse
        elif self.full:
            return False
        else:
            self.entries.append(impulse)
        self._upload()
        return True

    def clear(self) -> None:
        """Drop every queued impulse."""
        self.entries.clear()
        self._bodies.fill_(-1)

    def apply(self, state, body_com) -> None:
        """Launch the kernel that adds the queued impulses to ``state.body_f``; call it from the force callback."""
        wp.launch(
            _apply_impulses,
            dim=MAX_IMPULSES,
            inputs=[state.body_q, body_com, state.body_f, self._bodies, self._points, self._impulses, self._inverse_dt],
            device=self._device,
        )

    def forget_applied(self) -> None:
        """Drop the queue once the solver has applied it, which empties every slot on the device."""
        if self.entries and not (self._bodies.numpy() >= 0).any():
            self.entries.clear()

    def _upload(self) -> None:
        bodies = np.full(MAX_IMPULSES, -1, dtype=np.int32)
        points = np.zeros((MAX_IMPULSES, 3), dtype=np.float32)
        impulses = np.zeros((MAX_IMPULSES, 3), dtype=np.float32)
        for index, entry in enumerate(self.entries):
            bodies[index], points[index], impulses[index] = entry.body, entry.point, entry.vector
        self._points.assign(points)
        self._impulses.assign(impulses)
        self._bodies.assign(bodies)
