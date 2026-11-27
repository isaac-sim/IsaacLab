# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World-frame lidar hits drawn by Newton GL and Viser, and the default camera framing they share with Kit."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from isaaclab.utils.math import quat_apply
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg

if TYPE_CHECKING:
    from isaaclab_ov.renderers.ovrtx_lidar_sensor import OVRTXLidar

    from isaaclab.sim import SimulationContext
    from isaaclab.visualizers.base_visualizer import BaseVisualizer


LIDAR_POINT_RADIUS = 0.02
"""Sphere radius, in meters, used by viewers that log a point batch."""

LIDAR_POINT_COLOR = (1.0, 0.0, 0.0)
"""RGB color of logged lidar hits."""


def _iso8855_to_prim(points: torch.Tensor) -> torch.Tensor:
    """Map ISO 8855 sensor points onto the prim's camera axes.

    The beam travels along the prim's local ``-Z``. ``SENSOR`` returns use a different
    frame: ``+X`` is forward, ``+Y`` is left, and ``+Z`` is up. The pose quaternion
    rotates the prim axes, so a return has to be rewritten into those axes first.
    """
    forward, left, up = points.unbind(dim=-1)
    return torch.stack((-left, up, -forward), dim=-1)


def lidar_world_points(sensor: OVRTXLidar, env_ids: list[int] | None) -> np.ndarray:
    """Return finite lidar hits in world coordinates.

    ``WORLD`` returns are already in world coordinates. ``SENSOR`` returns are in the
    ISO 8855 sensor frame. They are rewritten onto the prim's camera axes, then
    rotated and translated by the pose the sensor writes to the renderer. Misses are
    omitted.

    Args:
        sensor: Lidar whose latest buffers are read.
        env_ids: Environments to include. None includes every environment. A global lidar
            has one buffer row and is not filtered by these IDs.

    Returns:
        Hit positions with shape ``(num_hits, 3)``. An empty cloud has shape ``(0, 3)``.
    """
    clouds = [cloud for _, cloud in lidar_env_hits(sensor, env_ids) if len(cloud)]
    if not clouds:
        return np.empty((0, 3), dtype=np.float32)
    if len(clouds) == 1:
        return clouds[0]
    return np.concatenate(clouds, axis=0)


def lidar_env_hits(sensor: OVRTXLidar, env_ids: list[int] | None) -> list[tuple[int, np.ndarray]]:
    """Return each environment's finite lidar hits in world coordinates.

    The transform matches :func:`lidar_world_points`. Environments are listed in ``env_ids`` order,
    or in scene order when ``env_ids`` is None. An environment with no finite hits is omitted.
    A global lidar has one buffer row, so visualized environment IDs do not index the scan.

    Args:
        sensor: Lidar whose latest buffers are read.
        env_ids: Environments to include. None includes every environment. A global lidar
            returns its single scan regardless of these IDs.

    Returns:
        ``(env_index, points)`` pairs. ``points`` has shape ``(num_hits, 3)``.
    """
    points = sensor.data.points.torch
    valid = sensor.data.valid.torch

    # A global lidar has one buffer row. Scene environment IDs therefore do not index that row.
    is_global = getattr(sensor, "_is_global", False)
    if is_global or env_ids is None:
        ids = list(range(points.shape[0]))
    else:
        ids = [int(env_id) for env_id in env_ids]
        if not ids:
            return []
        index = torch.as_tensor(ids, device=points.device, dtype=torch.long)
        points = points.index_select(0, index)
        valid = valid.index_select(0, index)

    if sensor.cfg.output_frame_of_reference == "SENSOR":
        position = sensor._pos_w
        rotation = sensor._quat_w
        if position is None or rotation is None:
            return []

        position = position.torch
        rotation = rotation.torch

        if env_ids is not None and not is_global:
            index = torch.as_tensor(ids, device=position.device, dtype=torch.long)
            position = position.index_select(0, index)
            rotation = rotation.index_select(0, index)

        points = quat_apply(rotation.unsqueeze(1), _iso8855_to_prim(points)) + position.unsqueeze(1)

    finite = valid & torch.isfinite(points).all(dim=-1)
    packed = points[finite].detach().to(dtype=torch.float32, device="cpu").numpy()
    counts = finite.sum(dim=-1).detach().to(device="cpu").numpy()
    offset = 0
    hits = []
    for env_id, count in zip(ids, counts, strict=True):
        count = int(count)
        if count:
            hits.append((env_id, packed[offset : offset + count]))
        offset += count
    return hits


def _as_vec3(values) -> tuple[float, float, float]:
    """Return three floats from a camera vector."""
    return (float(values[0]), float(values[1]), float(values[2]))


def _origins_cpu(origins) -> torch.Tensor | None:
    """Return environment origins as a CPU ``(num_envs, 3)`` tensor."""
    if origins is None:
        return None
    origin = torch.as_tensor(origins, dtype=torch.float32).detach()
    if origin.ndim != 2 or origin.shape[1] != 3 or origin.shape[0] == 0:
        return None
    return origin.to(device="cpu")


def frame_environments(eye, lookat, origins) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Back the camera up until every environment origin fits in view.

    The look target moves to the center of the origins and keeps its configured height. The eye
    keeps the configured view direction. A close-up of a single environment stays put. The fit
    uses a 30-degree half-angle, which covers a typical 3D view.

    Args:
        eye: Camera eye position, shape ``(3,)``.
        lookat: Camera look target, shape ``(3,)``.
        origins: Environment origins, shape ``(num_envs, 3)``.

    Returns:
        Eye and look target.
    """
    eye_t = torch.tensor(_as_vec3(eye), dtype=torch.float32)
    look_t = torch.tensor(_as_vec3(lookat), dtype=torch.float32)
    origin = _origins_cpu(origins)
    if origin is None:
        return _as_vec3(eye), _as_vec3(lookat)
    look = origin.mean(dim=0)
    look[2] = look_t[2]
    direction = eye_t - look_t
    span = torch.linalg.vector_norm(direction)
    if float(span) < 1e-4:
        direction = torch.tensor([1.0, -1.0, 0.5])
        span = torch.linalg.vector_norm(direction)
    direction = direction / span
    radius = torch.linalg.vector_norm(origin - look, dim=-1).amax()
    needed = float(radius) / math.tan(math.radians(30.0)) * 1.15
    distance = max(float(span), needed)
    return _as_vec3(look + direction * distance), _as_vec3(look)


def frame_visualizer_closeup(visualizer: BaseVisualizer, cameras) -> None:
    """Aim the default view so every environment origin fits in view.

    The visualizer configuration is the default view. A perspective camera that still has that
    pose was captured from it before initialization, so it receives the framed pose too. A camera
    with its own eye and look target keeps that angle. A partial environment subset keeps the
    configured camera.

    Args:
        visualizer: Visualizer whose configuration supplies the default camera.
        cameras: Cameras passed to the visualizer's ``initialize``.
    """
    if visualizer._env_ids is not None:
        return
    scene = visualizer._sim.get_scene_data_provider().get_interactive_scene()
    origins = None if scene is None else getattr(scene, "env_origins", None)
    configured = (_as_vec3(visualizer.cfg.eye), _as_vec3(visualizer.cfg.lookat))
    eye, lookat = frame_environments(*configured, origins)
    visualizer.cfg.eye = eye
    visualizer.cfg.lookat = lookat
    sources = list(cameras)
    for view in visualizer._image_views:
        source = getattr(getattr(view, "cfg", None), "source", None)
        if source is not None:
            sources.append(source)
    for camera in sources:
        if isinstance(camera, PerspectiveCameraCfg) and (_as_vec3(camera.eye), _as_vec3(camera.lookat)) == configured:
            camera.eye = eye
            camera.lookat = lookat


def _ovrtx_lidar_cls() -> type | None:
    """Return :class:`~isaaclab_ov.renderers.ovrtx_lidar_sensor.OVRTXLidar`, or None if it is unavailable."""
    try:
        from isaaclab_ov.renderers.ovrtx_lidar_sensor import OVRTXLidar
    except ImportError:
        return None
    return OVRTXLidar


def scene_has_lidar(sim: SimulationContext) -> bool:
    """Return whether the interactive scene owns an OvRTX lidar.

    Args:
        sim: Simulation whose interactive scene is searched.

    Returns:
        True when at least one sensor is an OvRTX lidar.
    """
    scene = sim.get_scene_data_provider().get_interactive_scene()
    lidar_cls = _ovrtx_lidar_cls()
    if scene is None or lidar_cls is None:
        return False
    return any(isinstance(sensor, lidar_cls) for sensor in scene.sensors.values())


def collect_lidar_hits(sim: SimulationContext, env_ids: list[int] | None) -> list[tuple[str, np.ndarray]]:
    """Return every visualized environment's lidar hits.

    ``env_ids`` None includes every environment. Each lidar is included even when its cloud is
    empty, so a viewer can hide the previous batch. Other sensors are ignored.

    Args:
        sim: Simulation whose interactive scene owns the lidars.
        env_ids: Visualized environments. None includes every environment.

    Returns:
        ``(sensor_name, points)`` pairs.
    """
    scene = sim.get_scene_data_provider().get_interactive_scene()
    lidar_cls = _ovrtx_lidar_cls()
    if scene is None or lidar_cls is None:
        return []

    selected = None if env_ids is None else list(env_ids)
    hits = []
    for name, sensor in scene.sensors.items():
        if isinstance(sensor, lidar_cls):
            hits.append((name, lidar_world_points(sensor, selected)))
    return hits


def sync_lidar_hits(
    sim: SimulationContext,
    env_ids: list[int] | None,
    viewer: Any,
    enabled: bool,
    previous_names: list[str],
) -> list[str]:
    """Upload lidar hits when enabled, or hide the previous batch when disabled.

    A disabled step with no previous names does not read the scene. A disabled step with previous
    names logs empty clouds so the last batch disappears.

    Args:
        sim: Simulation whose interactive scene owns the lidars.
        env_ids: Visualized environments. None includes every environment.
        viewer: Newton or Viser viewer.
        enabled: Whether lidar hits are copied and uploaded this step.
        previous_names: Sensor names logged on a previous enabled step.

    Returns:
        Sensor names logged this step, or an empty list when the cloud is hidden.
    """
    if not enabled:
        if previous_names:
            log_lidar_hits(viewer, [(name, np.empty((0, 3), dtype=np.float32)) for name in previous_names])
        return []
    hits = collect_lidar_hits(sim, env_ids)
    log_lidar_hits(viewer, hits)
    return [name for name, _ in hits]


def log_lidar_hits(viewer, hits: list[tuple[str, np.ndarray]], radius: float = LIDAR_POINT_RADIUS) -> None:
    """Log collected hits through a viewer ``log_points`` method.

    An empty cloud is logged as ``None``, which hides the previous batch. Newton GL's point kernel
    runs on the viewer device, so that batch and its per-point colors are allocated there. Viser
    receives a CPU batch and the shared color tuple.

    Args:
        viewer: Newton or Viser viewer.
        hits: Sensor names and world-frame clouds from :func:`collect_lidar_hits`.
        radius: Sphere radius in meters.
    """
    import warp as wp

    # ``NewtonViewerGL`` subclasses ``ViewerGL``. Both launch ``update_vbo_transforms_from_points``
    # on ``viewer.device`` and upload colors with ``colors.numpy()``.
    gl = type(viewer).__name__.endswith("GL")
    device = viewer.device if gl else "cpu"
    for name, points in hits:
        if len(points) == 0:
            cloud = None
            colors = LIDAR_POINT_COLOR
        else:
            cloud = wp.array(points, dtype=wp.vec3, device=device)
            if gl:
                colors = wp.full(len(points), wp.vec3(*LIDAR_POINT_COLOR), dtype=wp.vec3, device=device)
            else:
                colors = LIDAR_POINT_COLOR
        viewer.log_points(f"lidar/{name}", cloud, radii=radius, colors=colors)
