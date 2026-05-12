# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable description of a tiled camera passed to render backends."""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.sensors.camera.camera_cfg import CameraCfg


@dataclass(frozen=True)
class CameraRenderSpec:
    """Stable inputs for :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`.

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
