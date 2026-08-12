# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for visualizer and recorder camera image views."""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import numpy as np
import torch
import warp as wp

from pxr import Sdf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim.views import FrameView

_log = logging.getLogger(__name__)

_GENERATED_CAMERA_NAME = "VisualizerCamera"
VISUALIZER_TILED_CAMERA_MAX_TILES = 100

# Shared streaming camera registry.
# Key: (renderer_class_name, stage_path, num_envs, width, height, sorted_data_types)
# Maps the full camera configuration to a (Camera, generated_paths) tuple so that
# OVRTX (a process singleton — only one /Render prim allowed) is created once and
# reused by all visualizers that request the same configuration.
#
# A secondary dict maps (renderer_class_name, stage_path) → full_key so that a
# config mismatch for the same renderer on the same stage can be caught early
# with a descriptive error instead of hitting a USD /Render conflict.
_shared_streaming_cameras: dict[tuple, tuple] = {}
_renderer_stage_keys: dict[tuple[str, str], tuple] = {}


def resolve_tiled_env_indices(
    num_envs: int,
    streaming_envs: int | list[int],
    env_indices: list[int] | None,
    max_tiles: int | None = None,
    sample_from: list[int] | None = None,
) -> list[int]:
    """Resolve env ids for streaming camera view once at visualizer initialization."""
    if num_envs <= 0:
        return []
    _max_envs = len(streaming_envs) if isinstance(streaming_envs, list) else int(streaming_envs)
    if env_indices is not None:
        max_count = min(max(1, _max_envs), num_envs)
        if max_tiles is not None:
            max_count = min(max_count, max(1, int(max_tiles)))
        return [idx for idx in env_indices if 0 <= int(idx) < num_envs][:max_count]
    candidates = [
        idx for idx in (sample_from if sample_from is not None else range(num_envs)) if 0 <= int(idx) < num_envs
    ]
    if not candidates:
        return []
    max_count = min(max(1, _max_envs), len(candidates))
    if max_tiles is not None:
        max_count = min(max_count, max(1, int(max_tiles)))
    return sorted(random.sample(candidates, max_count))


def resolve_mono_env_index(num_envs: int) -> list[int]:
    """Return env_0 for mono sensor camera views."""
    return [0] if num_envs > 0 else []


def env_path_from_template(path_template: str, env_id: int) -> str:
    """Resolve common env wildcard/template spellings to a concrete env path."""
    path = path_template
    if "%d" in path:
        return path % env_id
    if "{}" in path:
        return path.format(env_id)
    path = path.replace("/World/envs/*", f"/World/envs/env_{env_id}")
    path = path.replace("/World/envs/env_.*", f"/World/envs/env_{env_id}")
    path = path.replace("/World/envs/env_.*/", f"/World/envs/env_{env_id}/")
    return path


def _camera_concrete_paths(camera: Camera) -> list[str]:
    view = getattr(camera, "_view", None)
    prims = getattr(view, "prims", None)
    if not prims:
        return []
    return [prim.GetPath().pathString for prim in prims]


def find_camera_by_prim_path(camera_sensors: dict[str, Camera], cam_prim_path: str, env_indices: list[int]) -> Camera:
    """Find a scene-owned Camera by config template or concrete camera prim paths."""
    wanted = {env_path_from_template(cam_prim_path, env_id) for env_id in env_indices}
    for camera in camera_sensors.values():
        if getattr(camera.cfg, "prim_path", None) == cam_prim_path:
            return camera
        concrete = set(_camera_concrete_paths(camera))
        if wanted and wanted.issubset(concrete):
            return camera
    stage = sim_utils.get_current_stage()
    stage_matches = [path for path in wanted if stage.GetPrimAtPath(path).IsValid()] if stage is not None else []
    if stage_matches:
        raise RuntimeError(
            f"cam_prim_path={cam_prim_path!r} matched USD camera prims, but no Isaac Lab Camera sensor owns them. "
            "Add the camera to scene.sensors or leave streaming_sensor_prim_path unset to use generated tiled cameras."
        )
    if not camera_sensors:
        raise RuntimeError(
            f"No Isaac Lab Camera sensors are registered in the scene, so streaming_sensor_prim_path={cam_prim_path!r} "
            "cannot be used. Use an environment that defines Camera sensors, or leave streaming_sensor_prim_path unset "
            "to use generated tiled cameras."
        )
    available_paths = {
        getattr(camera.cfg, "prim_path", None) for camera in camera_sensors.values() if getattr(camera, "cfg", None)
    }
    raise RuntimeError(
        f"No Isaac Lab Camera sensor matched cam_prim_path={cam_prim_path!r}. "
        f"Available Camera sensor prim paths: {sorted(path for path in available_paths if path)}. "
        "Leave streaming_sensor_prim_path unset to use generated tiled cameras."
    )


def ensure_camera_initialized(camera: Camera) -> None:
    """Initialize a visualizer-owned Camera created after the normal physics-ready callback."""
    if not camera.is_initialized:
        camera._initialize_callback(None)


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
    if not (math.isfinite(target_aspect) and target_aspect > 0):
        target_aspect = 1.0
    h, w = frames[0].shape[:2]
    env_cols = _best_streaming_cols(n_envs, n_gt, h, w, target_aspect=target_aspect)
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


def _best_streaming_cols(n_envs: int, n_gt: int, frame_h: int, frame_w: int, target_aspect: float = 1.0) -> int:
    """Env-column count that best matches the target composite aspect ratio.

    Prioritises complete rows (fewest empty cells) first, then minimises
    ``|log(composite_W/composite_H / target_aspect)|`` to match the window,
    with more columns as a final tiebreaker.

    Args:
        n_envs: Number of environment tiles.
        n_gt: Number of GT types per tile (columns per env).
        frame_h: Height of each tile in pixels.
        frame_w: Width of each tile in pixels.
        target_aspect: Desired composite width/height ratio (default: 1.0 square).
    """
    best_cols, best_score = 1, float("inf")
    for cols in range(1, n_envs + 1):
        rows = math.ceil(n_envs / cols)
        empty_cells = rows * cols - n_envs
        composite_w = cols * n_gt * frame_w
        composite_h = rows * frame_h
        # Distance from target aspect ratio: 0 is a perfect match.
        aspect_score = abs(math.log(composite_w / composite_h / target_aspect))
        # Strong penalty for ragged rows; break ties by aspect then prefer more cols.
        score = empty_cells * 10.0 + aspect_score - cols * 1e-6
        if score < best_score:
            best_score, best_cols = score, cols
    return best_cols


def create_visualizer_camera(
    *,
    num_envs: int,
    camera_name: str = _GENERATED_CAMERA_NAME,
    width: int,
    height: int,
    renderer_cfg: Any,
    data_types: list[str] | None = None,
    target_prim_path: str | None = None,
    eye: tuple[float, float, float] | None = None,
    streaming_envs: tuple[int, ...] | None = None,
) -> tuple[Camera, list[str], bool, tuple]:
    """Create an internal Camera sensor for visualizer image views.

    When the renderer is a singleton (e.g. OVRTX) a shared instance is returned
    on subsequent calls to avoid the ``/Render`` prim duplication error.  The
    cache is keyed by ``(renderer_class_name, stage_id, num_envs, width, height,
    sorted_data_types, target_prim_path, eye, streaming_envs)`` so that different
    stages (e.g. between tests) or different shape/type/env-index configurations
    never silently share an incompatible camera.

    .. note::
        The cache key includes ``streaming_envs`` (the resolved environment indices
        being streamed).  Two visualizers with identical renderer/resolution settings
        but different ``streaming_envs`` are treated as separate configurations; if
        both use the same singleton renderer they cannot coexist, and a warning is
        logged while the existing camera is returned as a non-owner.

    Args:
        num_envs: Total number of simulation environments.
        camera_name: USD prim name suffix for the auto-generated camera prims.
        width: Tile width in pixels.
        height: Tile height in pixels.
        renderer_cfg: Renderer configuration object; its class name is used as
            part of the cache key.
        data_types: Ground-truth data types to capture (e.g. ``["rgb"]``).
        target_prim_path: Optional USD path used as the camera's look-at target
            (included in the cache key for conflict detection).
        eye: Optional ``(x, y, z)`` eye offset in metres (included in the cache key).
        streaming_envs: Tuple of resolved environment indices that will be streamed.
            ``None`` is treated as an absent/unspecified selection and is stored as-is
            in the cache key.

    Returns:
        A 4-tuple ``(camera, generated_paths, is_owner, cache_key)`` where
        ``is_owner`` is ``True`` only for the first caller (who created the
        camera) and ``cache_key`` should be passed to :func:`evict_visualizer_camera`
        in the owner's ``close()`` path.

    Raises:
        RuntimeError: If the same renderer class is requested on the same stage
            with a different camera configuration (e.g. different ``num_envs`` or
            resolution), which would indicate a /Render prim singleton conflict.
    """
    stage = sim_utils.get_current_stage()
    # Use identifier (unique per anonymous stage) rather than realPath (always "" for in-memory stages).
    stage_id = stage.GetRootLayer().identifier if stage is not None else ""
    renderer_class = type(renderer_cfg).__name__
    dt_key = tuple(sorted(data_types or ["rgb"]))
    eye_key = tuple(float(x) for x in eye) if eye is not None else None
    # Normalise streaming_envs so that callers passing lists or arrays produce the same key.
    envs_key = tuple(int(i) for i in streaming_envs) if streaming_envs is not None else None
    full_key = (
        renderer_class,
        stage_id,
        int(num_envs),
        int(width),
        int(height),
        dt_key,
        target_prim_path,
        eye_key,
        envs_key,
    )
    class_stage_key = (renderer_class, stage_id)

    if full_key in _shared_streaming_cameras:
        camera, generated_paths = _shared_streaming_cameras[full_key]
        return camera, generated_paths, False, full_key

    # Same renderer class on the same stage but a different configuration → potential conflict.
    if class_stage_key in _renderer_stage_keys and _renderer_stage_keys[class_stage_key] != full_key:
        prev = _renderer_stage_keys[class_stage_key]
        # Check whether the configurations differ only in streaming_envs (index 8).
        # A streaming_envs-only mismatch means two visualizers want different env subsets
        # on a singleton renderer — we cannot satisfy both, so we return the existing camera
        # as a non-owner with a warning instead of hard-failing.
        prev_without_envs = prev[:8]
        req_without_envs = full_key[:8]
        if prev_without_envs == req_without_envs:
            camera, generated_paths = _shared_streaming_cameras[prev]
            _log.warning(
                "Two visualizers share renderer '%s' on the same stage but requested different "
                "streaming_envs (%s vs %s). The singleton renderer cannot produce two independent "
                "views; the second visualizer will observe the first visualizer's camera frames. "
                "Ensure all visualizers that share this renderer use the same streaming_envs.",
                renderer_class,
                list(prev[8]) if prev[8] is not None else None,
                list(envs_key) if envs_key is not None else None,
            )
            return camera, generated_paths, False, prev
        raise RuntimeError(
            f"Cannot create a second streaming camera with renderer '{renderer_class}' on the same stage "
            f"using a different configuration. The renderer is a process singleton.\n"
            f"  Existing : num_envs={prev[2]}, width={prev[3]}, height={prev[4]}, data_types={list(prev[5])}\n"
            f"  Requested: num_envs={num_envs}, width={width}, height={height}, data_types={list(dt_key)}\n"
            "Ensure all visualizers that share this renderer use the same streaming camera settings."
        )

    spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    )
    generated_paths = [f"/World/envs/env_{env_id}/{camera_name}" for env_id in range(int(num_envs))]
    for path in generated_paths:
        if len(sim_utils.find_matching_prims(path)) == 0:
            spawn.func(path, spawn, translation=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0))

    stage = sim_utils.get_current_stage()
    for path in generated_paths:
        cam_prim = stage.GetPrimAtPath(path)
        attr = cam_prim.GetAttribute("omni:scenePartition")
        if not attr.IsValid():
            attr = cam_prim.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token)
        attr.Set(path.split("/")[-2])
    cfg = CameraCfg(
        prim_path=f"/World/envs/env_.*/{camera_name}",
        update_period=0.0,
        height=int(height),
        width=int(width),
        data_types=data_types if data_types is not None else ["rgb"],
        spawn=None,
        renderer_cfg=renderer_cfg,
    )
    camera = Camera(cfg)
    ensure_camera_initialized(camera)
    _shared_streaming_cameras[full_key] = (camera, generated_paths)
    _renderer_stage_keys[class_stage_key] = full_key
    return camera, generated_paths, True, full_key


def evict_visualizer_camera(key: tuple | None) -> None:
    """Remove a cached streaming camera entry from the shared registry.

    Call from a visualizer's ``close()`` path when it owns the camera so that
    subsequent tests or sessions can create a fresh camera rather than receiving
    a stale object whose USD prims no longer exist.

    Args:
        key: The cache key returned as the fourth element of
            :func:`create_visualizer_camera`, or ``None`` (no-op).
    """
    if key is None:
        return
    _shared_streaming_cameras.pop(key, None)
    class_stage_key = (key[0], key[1])
    if _renderer_stage_keys.get(class_stage_key) == key:
        _renderer_stage_keys.pop(class_stage_key, None)


def remove_generated_prims(prim_paths: list[str] | None) -> None:
    """Remove visualizer-owned camera prims from the current stage and clear their cache entries.

    Any entry in :data:`_shared_streaming_cameras` whose ``generated_paths`` list overlaps
    with ``prim_paths`` is evicted so that a subsequent :func:`create_visualizer_camera` call
    can create a fresh camera rather than returning a stale object whose USD prims no longer exist.
    """
    if not prim_paths:
        return

    # Evict cache entries that owned any of the prims being removed.
    prim_path_set = set(prim_paths)
    stale_keys = [
        key for key, (_, gen_paths) in list(_shared_streaming_cameras.items()) if prim_path_set.intersection(gen_paths)
    ]
    for key in stale_keys:
        evict_visualizer_camera(key)

    stage = sim_utils.get_current_stage()
    if stage is None:
        return
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)


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


def compute_tile_resolution(window_width: int, window_height: int, num_tiles: int, n_gt: int = 1) -> tuple[int, int]:
    """Derive a conservative per-tile resolution from the visualizer window.

    Args:
        window_width: Visualizer window width in pixels.
        window_height: Visualizer window height in pixels.
        num_tiles: Number of environment tiles (one per env).
        n_gt: Number of GT data types per tile (e.g. 2 for rgb+depth).  The composite
            image is ``env_cols * n_gt`` tiles wide, so each tile gets
            ``window_width // (env_cols * n_gt)`` pixels of horizontal space.

    Returns:
        Per-tile ``(width, height)`` in pixels.
    """
    if window_width <= 0 or window_height <= 0:
        raise ValueError(f"Window dimensions must be positive, got {window_width}x{window_height}.")
    n_gt = max(1, int(n_gt))
    cols = max(1, math.ceil(math.sqrt(max(1, num_tiles))))
    rows = math.ceil(max(1, num_tiles) / cols)
    return max(1, int(window_width) // (cols * n_gt)), max(1, int(window_height) // rows)


def _normalize_env0_path(path_template: str) -> str:
    """Resolve env template spellings to env_0 for path comparison."""
    return env_path_from_template(path_template, 0)


def _scene_articulation_positions(scene: Any, prim_path_template: str, env_indices: list[int]) -> torch.Tensor | None:
    """Resolve follow positions from scene articulation state when the path targets an asset/body."""
    follow_env0 = _normalize_env0_path(prim_path_template)
    for asset in getattr(scene, "articulations", {}).values():
        asset_path = _normalize_env0_path(getattr(asset.cfg, "prim_path", ""))
        if not asset_path:
            continue
        if follow_env0 == asset_path:
            return asset.data.root_pos_w.torch[env_indices].detach().cpu()
        prefix = asset_path + "/"
        if not follow_env0.startswith(prefix):
            continue
        body_name = follow_env0.removeprefix(prefix).split("/")[-1]
        if body_name not in asset.body_names:
            continue
        body_ids, _ = asset.find_bodies(body_name)
        if not body_ids:
            continue
        return asset.data.body_pos_w.torch[env_indices, int(body_ids[0])].detach().cpu()
    return None


def prim_world_positions(
    stage: Any, prim_path_template: str, env_indices: list[int], scene: Any | None = None
) -> torch.Tensor:
    """Return world-space translations for concrete prim paths resolved from env ids.

    Uses scene articulation state first when the target is an asset/body path,
    then falls back to ``FrameView`` and USD for arbitrary prim paths.
    """
    if scene is not None:
        positions_tensor = _scene_articulation_positions(scene, prim_path_template, env_indices)
        if positions_tensor is not None:
            return positions_tensor

    xform_cache = UsdGeom.XformCache()
    positions = []
    try:
        for env_id in env_indices:
            prim_path = env_path_from_template(prim_path_template, env_id)
            view = FrameView(prim_path, device="cpu", stage=stage)
            try:
                if view.count != 1:
                    raise RuntimeError(f"expected one prim, got {view.count}")
                pos_w, _ = view.get_world_poses()
                pos = pos_w.torch[0].detach().cpu()
            finally:
                view.close()
            positions.append((float(pos[0]), float(pos[1]), float(pos[2])))
        return torch.tensor(positions, dtype=torch.float32)
    except Exception:
        positions.clear()

    for env_id in env_indices:
        prim_path = env_path_from_template(prim_path_template, env_id)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"streaming_cam_target_prim_path resolved to missing prim: {prim_path!r}.")
        transform = xform_cache.GetLocalToWorldTransform(prim)
        translation = transform.ExtractTranslation()
        positions.append((float(translation[0]), float(translation[1]), float(translation[2])))
    return torch.tensor(positions, dtype=torch.float32)


def apply_camera_view_from_origins(
    camera: Camera,
    origins: torch.Tensor,
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
    env_ids: list[int] | None = None,
) -> None:
    """Set camera poses from origins plus relative eye/lookat offsets."""
    device = camera.device
    origins = origins.to(device=device)
    eye_offset = torch.tensor(eye, dtype=torch.float32, device=device).unsqueeze(0)
    lookat_offset = torch.tensor(lookat, dtype=torch.float32, device=device).unsqueeze(0)
    camera.set_world_poses_from_view(origins + eye_offset, origins + lookat_offset, env_ids=env_ids)
    camera._update_poses(None)


def apply_camera_target_positions(
    camera: Camera,
    target_positions: torch.Tensor,
    eye: tuple[float, float, float],
    env_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set generated tiled camera poses as target-relative eye offsets."""
    device = camera.device
    target_positions = target_positions.to(device=device)
    eye_offset = torch.tensor(eye, dtype=torch.float32, device=device).unsqueeze(0)
    eyes = target_positions + eye_offset
    targets = target_positions
    camera.set_world_poses_from_view(eyes, targets, env_ids=env_ids)
    camera._update_poses(None)
    return eyes.detach().cpu(), targets.detach().cpu()
