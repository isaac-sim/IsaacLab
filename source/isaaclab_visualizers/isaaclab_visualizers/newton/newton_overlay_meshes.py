# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw Newton's line and arrow overlays in the OVRTX view as thin meshes.

Newton's viewer builds its debug overlays (joints, contacts, springs, centers of mass, inertia boxes) from
:meth:`log_lines` and :meth:`log_arrows` calls that only the OpenGL viewer draws. :class:`OverlayMeshPublisher`
records those calls and republishes each batch as world-space meshes through
:meth:`~isaaclab_ov.renderers.OVRTXRenderer.log_mesh`, so the OVRTX view shows the same overlays.
"""

from __future__ import annotations

import numpy as np
import warp as wp

_VERTICES_PER_SEGMENT = 13
"""Four shaft corners at each end, four arrowhead base corners, and the arrowhead tip."""

_MAX_COLORS = 6
"""Most distinct colors drawn per batch; further colors take the first color."""

_COLOR_REFRESH_INTERVAL = 60
"""Publishes between reads of a batch's per-segment colors from the device."""

DEFAULT_COLOR = (0.8, 0.8, 0.8)
"""Color of a batch that supplies none."""

LINE_RADIUS = 0.004
"""Radius of a line segment [m]."""

_SEGMENT_TRIANGLES = (
    # shaft sides: corner k of the start ring, corner k of the end ring
    *((k, (k + 1) % 4, 4 + (k + 1) % 4) for k in range(4)),
    *((k, 4 + (k + 1) % 4, 4 + k) for k in range(4)),
    # shaft start cap
    (0, 2, 1),
    (0, 3, 2),
    # arrowhead base
    (8, 9, 10),
    (8, 10, 11),
    # arrowhead sides
    *((8 + k, 8 + (k + 1) % 4, 12) for k in range(4)),
)


@wp.kernel
def build_segment_meshes(
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
    classes: wp.array[wp.int32],
    class_id: int,
    shaft_radius: float,
    head_radius: float,
    head_length: float,
    points: wp.array[wp.vec3],
):
    """Write the vertices of one thin box (and arrowhead) per segment; other classes collapse to a point."""
    i = wp.tid()
    base = i * 13
    start = starts[i]
    direction = ends[i] - start
    length = wp.length(direction)
    if classes[i] != class_id or length < 1.0e-6:
        for k in range(13):
            points[base + k] = start
        return
    axis = direction / length
    helper = wp.vec3(0.0, 0.0, 1.0)
    if wp.abs(axis[2]) > 0.9:
        helper = wp.vec3(1.0, 0.0, 0.0)
    u = wp.normalize(wp.cross(axis, helper))
    v = wp.cross(axis, u)
    head = wp.min(head_length, 0.5 * length)
    shaft_end = ends[i] - axis * head
    for k in range(4):
        # corners (-1, -1), (1, -1), (1, 1), (-1, 1)
        cx = float(1.0)
        if k == 0 or k == 3:
            cx = -1.0
        cy = float(1.0)
        if k < 2:
            cy = -1.0
        offset = u * (cx * shaft_radius) + v * (cy * shaft_radius)
        points[base + k] = start + offset
        points[base + 4 + k] = shaft_end + offset
        points[base + 8 + k] = shaft_end + u * (cx * head_radius) + v * (cy * head_radius)
    points[base + 12] = ends[i]


def segment_indices(count: int, device) -> wp.array:
    """Return the triangle indices of *count* segments built by :func:`build_segment_meshes`."""
    template = np.array(_SEGMENT_TRIANGLES, np.int32).reshape(-1)
    indices = (template[None, :] + _VERTICES_PER_SEGMENT * np.arange(count, dtype=np.int32)[:, None]).reshape(-1)
    return wp.array(indices, dtype=wp.int32, device=device)


def color_classes(colors, count: int) -> tuple[np.ndarray | None, list[tuple[float, float, float]]]:
    """Group a batch's colors into at most :data:`_MAX_COLORS` distinct colors.

    Args:
        colors: A color triplet, a Warp array with a color per segment, or ``None``.
        count: Number of segments in the batch.

    Returns:
        The class index of each segment (``None`` when the batch has one color) and the color of each class.
    """
    if colors is None:
        return None, [DEFAULT_COLOR]
    if isinstance(colors, (tuple, list)):
        return None, [tuple(float(value) for value in colors[:3])]
    values = colors.numpy()
    if values.size == 3 * count:
        values = values.reshape(count, 3)
    if values.ndim != 2 or values.shape[0] != count or values.shape[1] != 3:
        return None, [DEFAULT_COLOR]
    rounded = np.round(values, 3)
    unique, inverse, frequency = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-frequency)[:_MAX_COLORS]
    rank = np.zeros(len(unique), np.int32)  # colors beyond the cap take the most frequent one
    rank[order] = np.arange(len(order), dtype=np.int32)
    return rank[inverse.reshape(-1)].astype(np.int32), [tuple(float(c) for c in unique[i]) for i in order]


class _Batch:
    """Meshes of one named line or arrow batch, rebuilt only when its segment count changes."""

    def __init__(self, count: int, device) -> None:
        self.count = count
        self.indices = segment_indices(count, device)
        self.points: list[wp.array] = []
        self.colors: list[tuple[float, float, float]] = []
        self.classes: wp.array | None = None
        self.publishes = 0


class OverlayMeshPublisher:
    """Record Newton's line and arrow overlay calls and publish them as meshes to OVRTX renderers."""

    def __init__(self, prefix: str) -> None:
        """Create a publisher.

        Args:
            prefix: Mesh path prefix, unique to the owning visualizer.
        """
        self._prefix = prefix
        self._frame: dict[str, tuple] = {}
        self._batches: dict[str, _Batch] = {}
        self._published: dict[str, list[str]] = {}

    def record(self, name, starts, ends, colors, width: float, hidden: bool, arrow: bool) -> None:
        """Remember one overlay batch of the current frame, as Newton's viewer receives it."""
        self._frame[name] = (starts, ends, colors, width, hidden, arrow)

    def publish(self, renderers) -> None:
        """Publish the batches recorded since the last call, and hide every earlier batch that was not recorded."""
        frame, self._frame = self._frame, {}
        for name in {*self._published, *frame}:
            batch = frame.get(name)
            if batch is None or batch[4] or batch[0] is None or batch[1] is None or len(batch[0]) == 0:
                self._hide(name, renderers)
            else:
                self._show(name, *batch, renderers)

    def hide_all(self, renderers) -> None:
        """Hide every published overlay mesh."""
        self._frame.clear()
        for name in tuple(self._published):
            self._hide(name, renderers)

    def _path(self, name: str, index: int, count: int) -> str:
        return f"{self._prefix}/overlay/{name.strip('/').replace('/', '_')}_{count}_{index}"

    def _show(self, name, starts, ends, colors, width, _hidden, arrow, renderers) -> None:
        count = len(starts)
        batch = self._batches.get(name)
        if batch is None or batch.count != count:
            self._hide(name, renderers)
            batch = self._batches[name] = _Batch(count, starts.device)
        if batch.classes is None or batch.publishes % _COLOR_REFRESH_INTERVAL == 0:
            classes, batch.colors = color_classes(colors, count)
            batch.classes = wp.array(
                np.zeros(count, np.int32) if classes is None else classes, dtype=wp.int32, device=starts.device
            )
            while len(batch.points) < len(batch.colors):
                batch.points.append(wp.zeros(count * _VERTICES_PER_SEGMENT, dtype=wp.vec3, device=starts.device))
        batch.publishes += 1
        shaft = max(0.5 * float(width), LINE_RADIUS) if arrow else LINE_RADIUS
        head_radius, head_length = (2.5 * shaft, 6.0 * shaft) if arrow else (0.0, 0.0)
        paths = []
        for index, color in enumerate(batch.colors):
            points = batch.points[index]
            wp.launch(
                build_segment_meshes,
                count,
                [starts, ends, batch.classes, index, shaft, head_radius, head_length, points],
                device=starts.device,
            )
            path = self._path(name, index, count)
            paths.append(path)
            for renderer in renderers:
                renderer.log_mesh(
                    path, points, batch.indices, hidden=False, backface_culling=False, color=color,
                    roughness=0.8, metallic=0.0,
                )  # fmt: skip
        self._published[name] = paths

    def _hide(self, name: str, renderers) -> None:
        batch = self._batches.get(name)
        for index, path in enumerate(self._published.pop(name, ())):
            for renderer in renderers:
                renderer.log_mesh(path, batch.points[index], batch.indices, hidden=True)
