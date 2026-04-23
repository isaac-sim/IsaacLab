# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable description of a tiled camera passed to render backends."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors.sensor_base import SensorBase


@dataclass(frozen=True)
class CameraRenderSpec:
    """Stable inputs for :meth:`~isaaclab.renderers.base_renderer.BaseRenderer._create_render_data_impl`.

    Backends use this instead of holding a reference to the :class:`~isaaclab.sensors.camera.Camera`
    sensor instance, avoiding circular dependencies between sensors and render data.

    Args:
        cfg: Camera configuration (data types, resolution, filters, etc.).
        device: Torch device string (e.g. ``"cuda:0"``) used by GPU annotators and Warp.
        num_instances: Number of tiled camera instances (environments).
        camera_prim_paths: Absolute USD paths for each environment's camera prim.
        view_count: Number of camera prims (must match ``len(camera_prim_paths)``).
        camera_path_relative_to_env_0: Camera prim path with ``/World/envs/env_0/`` prefix
            stripped; required by OVRTX. Empty string if the first camera is not under env 0.
    """

    cfg: CameraCfg
    device: str
    num_instances: int
    camera_prim_paths: tuple[str, ...]
    view_count: int
    camera_path_relative_to_env_0: str

    @classmethod
    def coerce(cls, source: CameraRenderSpec | SensorBase) -> CameraRenderSpec:
        """Return a :class:`CameraRenderSpec`, warning if a sensor is passed (deprecated)."""
        if isinstance(source, CameraRenderSpec):
            return source
        warnings.warn(
            "Passing a sensor to BaseRenderer.create_render_data is deprecated; pass CameraRenderSpec instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return cls._from_sensor(source)

    @classmethod
    def _from_sensor(cls, sensor: SensorBase) -> CameraRenderSpec:
        """Build a spec from a camera-like sensor (internal / deprecated compatibility)."""
        view = getattr(sensor, "_view", None)
        if view is None:
            raise TypeError("Sensor has no _view; cannot build CameraRenderSpec.")
        paths = tuple(p.GetPath().pathString for p in view.prims)
        env_0_prefix = "/World/envs/env_0/"
        rel = paths[0].removeprefix(env_0_prefix) if paths[0].startswith(env_0_prefix) else ""
        dev = sensor.device
        device_str = dev if isinstance(dev, str) else str(dev)
        return cls(
            cfg=sensor.cfg,  # type: ignore[arg-type]
            device=device_str,
            num_instances=sensor.num_instances,
            camera_prim_paths=paths,
            view_count=view.count,
            camera_path_relative_to_env_0=rel,
        )
