# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for visualizer and recorder camera image views."""

from __future__ import annotations

import math
import random
import re
from typing import TYPE_CHECKING

from ...cloner.cloner_cfg import DEFAULT_ENV_TEMPLATE, expand_env_regex_ns
from ...visualizers.visualizer_cfg import PerspectiveCameraCfg, SceneCameraCfg

if TYPE_CHECKING:
    from ...sensors.camera import Camera
    from ...visualizers.visualizer_cfg import VisualizerCfg

VISUALIZER_TILED_CAMERA_MAX_TILES = 100


_CAMERA_CHANNEL_KEYS = {
    "rgb": ("rgb", "rgba"),
    "depth": ("depth", "distance_to_image_plane"),
    "segmentation": ("semantic_segmentation",),
    "normals": ("normals",),
}


def sensor_key_for_gt_type(
    gt_type: str, available_keys: frozenset[str] | None = None, *, required: bool = True
) -> str | None:
    """Bind a display channel to its available sensor output.

    Args:
        gt_type: Display channel: rgb, depth, normals, or segmentation.
        available_keys: Sensor output names. None returns the primary output name.
        required: Whether a missing output raises; False skips incompatible automatic sources.

    Returns:
        The matching sensor key, or None when absent and not required.

    Raises:
        ValueError: If the display channel is unknown.
        KeyError: If no matching output is available and required is True.
    """
    if gt_type not in _CAMERA_CHANNEL_KEYS:
        raise ValueError(f"GT type {gt_type!r} is not supported. Valid types: {sorted(_CAMERA_CHANNEL_KEYS)}")
    keys = _CAMERA_CHANNEL_KEYS[gt_type]
    if available_keys is None:
        return keys[0]
    for key in keys:
        if key in available_keys:
            return key
    if not required:
        return None
    raise KeyError(f"No sensor output found for GT type {gt_type!r}. Tried {keys}; available: {sorted(available_keys)}")


def resolve_camera_sources(
    cfg: VisualizerCfg, cameras: dict[str, Camera], *, env_template: str = DEFAULT_ENV_TEMPLATE
) -> list[PerspectiveCameraCfg | Camera]:
    """Bind display sources before initializing a visualizer, without reading sensor frames.

    Args:
        cfg: Requested camera sources and display channels. The configuration is not modified.
        cameras: Scene-owned sensors keyed by scene name.
        env_template: Environment namespace used to expand ``{ENV_REGEX_NS}`` references.

    Returns:
        Ordered perspective settings and borrowed sensors. Explicit scene references must support
        every requested channel; automatic discovery skips incompatible sensors.
    """
    sources = list(cfg.cameras or [PerspectiveCameraCfg(eye=cfg.eye, lookat=cfg.lookat, focal_length=cfg.focal_length)])
    if not cfg.streaming_view:
        return sources
    gt_types = cfg.streaming_gt_types
    for gt_type in gt_types:
        sensor_key_for_gt_type(gt_type)
    if cfg.cameras is None:
        if cfg.streaming_sensor_prim_path is not None:
            sources.insert(0, SceneCameraCfg(prim_path=cfg.streaming_sensor_prim_path))
        else:
            for camera in cameras.values():
                available = frozenset(camera.cfg.data_types)
                if all(sensor_key_for_gt_type(gt, available, required=False) is not None for gt in gt_types):
                    sources.append(camera)
    for index, source in enumerate(sources):
        if not isinstance(source, SceneCameraCfg):
            continue
        path = expand_env_regex_ns(source.prim_path, env_template)
        pattern = path.replace("%d", "[^/]+").replace("{}", "[^/]+")
        pattern = pattern.replace("/World/envs/*", "/World/envs/env_[^/]+")
        for camera in cameras.values():
            if camera.cfg.prim_path == path or (
                camera._view is not None
                and any(re.fullmatch(pattern, str(prim.GetPath())) for prim in camera._view.prims)
            ):
                break
        else:
            available_paths = sorted(camera.cfg.prim_path for camera in cameras.values())
            raise ValueError(
                f"No scene Camera matches prim_path={path!r}. "
                f"Declare a CameraCfg in the scene; available paths: {available_paths}."
            )
        available = frozenset(camera.cfg.data_types)
        for gt_type in gt_types:
            sensor_key_for_gt_type(gt_type, available)
        sources[index] = camera
    return sources


def resolve_streaming_envs(
    num_envs: int,
    streaming_envs: int | list[int],
    max_tiles: int = VISUALIZER_TILED_CAMERA_MAX_TILES,
    sample_from: list[int] | None = None,
) -> list[int]:
    """Resolve ``streaming_envs`` to a concrete list of env indices.

    Args:
        num_envs: Total number of simulation environments.
        streaming_envs: ``int`` → randomly sample that many envs;
            ``list[int]`` → use exactly those indices (capped at ``max_tiles``).
        max_tiles: Hard cap on the number of returned indices.
        sample_from: When ``streaming_envs`` is an ``int``, sample from this
            subset rather than all envs (e.g. visible env indices).

    Returns:
        Sorted list of env indices, length ≤ ``max_tiles``.
    """
    if isinstance(streaming_envs, list):
        indices = [i for i in streaming_envs if 0 <= i < num_envs]
        return sorted(indices[:max_tiles])
    pool = sample_from if sample_from is not None else list(range(num_envs))
    count = min(int(streaming_envs), max_tiles, len(pool))
    return sorted(random.sample(pool, count))


def image_grid_columns(n_envs: int, n_gt: int, height: int, width: int, target_aspect: float = 1.0) -> int:
    """Choose complete environment rows first, then the closest display aspect ratio."""
    if not (math.isfinite(target_aspect) and target_aspect > 0):
        target_aspect = 1.0
    best_cols, best_score = 1, float("inf")
    for columns in range(1, n_envs + 1):
        rows = math.ceil(n_envs / columns)
        empty = rows * columns - n_envs
        aspect = columns * n_gt * width / (rows * height)
        score = empty * 10.0 + abs(math.log(aspect / target_aspect)) - columns * 1e-6
        if score < best_score:
            best_cols, best_score = columns, score
    return best_cols
