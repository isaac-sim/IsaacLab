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

import numpy as np
import torch
import warp as wp

from ...cloner.cloner_cfg import DEFAULT_ENV_TEMPLATE, expand_env_regex_ns
from ...visualizers.visualizer_cfg import PerspectiveCameraCfg, SceneCameraCfg
from .camera_colorizer import sensor_key_for_gt_type

if TYPE_CHECKING:
    from ...sensors.camera import Camera
    from ...visualizers.visualizer_cfg import VisualizerCfg

VISUALIZER_TILED_CAMERA_MAX_TILES = 100


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


def camera_gt_batch(camera: Camera, env_indices: list[int], sensor_key: str) -> torch.Tensor:
    """Return GT output for selected env indices from a camera sensor.

    Args:
        camera: Isaac Lab :class:`~isaaclab.sensors.camera.Camera` sensor.
        env_indices: Env indices to select (must be valid indices into the
            camera's tiled output).
        sensor_key: Key in ``camera.data.output``, e.g. ``"rgb"``,
            ``"depth"``, or ``"semantic_segmentation"``.

    Returns:
        Tensor of shape ``(len(env_indices), H, W, C)`` on the camera's device.
    """
    raw = camera.data.output[sensor_key]
    if isinstance(raw, wp.array):
        raw = wp.to_torch(raw)
    elif hasattr(raw, "torch"):
        raw = raw.torch
    if env_indices:
        idx = torch.tensor(env_indices, dtype=torch.long, device=raw.device)
        return raw.index_select(0, idx)
    return raw


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


def compose_streaming_grid(
    frames: list[np.ndarray],
    n_envs: int,
    n_gt: int,
    target_aspect: float = 1.0,
) -> np.ndarray:
    """Composite streaming frames into a tiled output image.

    Layout minimises ``|log(composite_W/composite_H / target_aspect)|`` subject
    to the constraint that all GT columns for one env remain on the same row.
    Pass ``target_aspect=window_width/window_height`` to fill the panel optimally.

    Args:
        frames: Flat list of ``uint8 (H, W, 3)`` arrays ordered as
            ``[env0_gt0, env0_gt1, ..., env0_gtM-1, env1_gt0, ...]``.
        n_envs: Number of environments represented in ``frames``.
        n_gt: Number of GT types per environment.
        target_aspect: Desired width-to-height ratio for the composite image.
            Defaults to ``1.0`` (square).  Use ``window_width / window_height``
            to fill the visualizer panel.  Must be positive and finite; invalid
            values (zero, negative, NaN, inf) fall back to ``1.0``.

    Returns:
        Single ``uint8 (total_H, total_W, 3)`` composite image, or a 1×1 black
        pixel if ``frames`` is empty.
    """
    if not frames:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h, w = frames[0].shape[:2]
    env_cols = image_grid_columns(n_envs, n_gt, h, w, target_aspect)
    env_rows = math.ceil(n_envs / env_cols)
    canvas = np.zeros((env_rows * h, env_cols * n_gt * w, 3), dtype=np.uint8)
    for env_idx in range(n_envs):
        ec = env_idx % env_cols
        er = env_idx // env_cols
        for gt_idx in range(n_gt):
            frame = frames[env_idx * n_gt + gt_idx]
            y0, x0 = er * h, (ec * n_gt + gt_idx) * w
            canvas[y0 : y0 + h, x0 : x0 + w] = frame[..., :3]
    return canvas


def camera_rgb_batch(camera: Camera, env_indices: list[int]) -> torch.Tensor:
    """Return RGB output for selected env indices."""
    rgb = camera.data.output["rgb"]
    if isinstance(rgb, wp.array):
        rgb = wp.to_torch(rgb)
    elif hasattr(rgb, "torch"):
        rgb = rgb.torch
    if env_indices:
        index = torch.tensor(env_indices, dtype=torch.long, device=rgb.device)
        return rgb.index_select(0, index)
    return rgb


def compose_rgb_grid_tensor(rgb_batch: torch.Tensor) -> torch.Tensor:
    """Compose an RGB batch into a near-square uint8 image grid without leaving its device."""
    if rgb_batch.ndim == 3:
        return rgb_batch[..., :3].contiguous()
    n, h, w, _ = rgb_batch.shape
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    rgb = rgb_batch[..., :3]
    pad = rows * cols - n
    if pad > 0:
        rgb = torch.cat([rgb, torch.zeros((pad, h, w, 3), dtype=rgb.dtype, device=rgb.device)], dim=0)
    return rgb.reshape(rows, cols, h, w, 3).permute(0, 2, 1, 3, 4).reshape(rows * h, cols * w, 3).contiguous()
