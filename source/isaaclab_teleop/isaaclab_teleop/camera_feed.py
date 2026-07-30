# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lifecycle management for camera image panels in XR."""

from __future__ import annotations

import importlib
import logging
import math
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch

from .isaac_teleop_cfg import XrCameraFeedCfg, XrCameraFeedLayoutCfg

if TYPE_CHECKING:
    from isaaclab.sensors import Camera

logger = logging.getLogger(__name__)

_RESPONSIVE_DENOISING_MIN_ISAAC_SIM_VERSION = (6, 1)


@lru_cache(maxsize=1)
def _camera_type() -> type[Camera]:
    """Load the concrete camera type only when feeds are presented."""
    from isaaclab.sensors import Camera

    return Camera


def _load_kit_scene_ui_presenter() -> Any | None:
    """Load the optional Kit SceneUI presenter without making Kit a package dependency."""
    try:
        module = importlib.import_module(".camera_feed_kit_scene_ui", __package__)
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "XR camera PiP is unavailable because Kit SceneUI could not be loaded (%s: %s). Continuing without PiP.",
            type(exc).__name__,
            exc,
        )
        return None
    return module._KitSceneUiCameraFeedPresenter()


def _configure_ray_reconstruction_compatibility(camera_cfg: Any, camera_name: str) -> None:
    """Fall back to classic DLSS on runtimes without responsive denoising."""
    renderer_cfg = getattr(camera_cfg, "renderer_cfg", None)
    if getattr(renderer_cfg, "enable_dlss_ray_reconstruction", None) is not True:
        return

    from isaaclab.utils.version import get_isaac_sim_version

    isaac_sim_version = get_isaac_sim_version()
    if (isaac_sim_version.major, isaac_sim_version.minor) < _RESPONSIVE_DENOISING_MIN_ISAAC_SIM_VERSION:
        renderer_cfg.enable_dlss_ray_reconstruction = False
        logger.warning(
            "XR camera feed %r requested DLSS Ray Reconstruction, but Isaac Sim %s predates responsive "
            "denoising. Falling back to classic DLSS.",
            camera_name,
            isaac_sim_version,
        )


def _prepare_camera_feed_cfgs(env_cfg: Any, cfgs: list[XrCameraFeedCfg]) -> list[XrCameraFeedCfg]:
    """Validate selected scene cameras."""
    from isaaclab.sensors import CameraCfg

    prepared: list[XrCameraFeedCfg] = []
    enabled_cfgs = deepcopy(cfgs)
    duplicates = sorted(name for name, count in Counter(cfg.camera_name for cfg in enabled_cfgs).items() if count > 1)
    if duplicates:
        raise ValueError(f"XR camera feeds must have unique camera names. Duplicates: {duplicates}.")

    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        raise ValueError("XR camera feeds require an environment configuration with a scene.")
    for cfg in enabled_cfgs:
        camera_cfg = getattr(scene, cfg.camera_name, None)
        if camera_cfg is None:
            raise ValueError(f"XR camera feed {cfg.camera_name!r} is not present in the scene.")
        if not isinstance(camera_cfg, CameraCfg):
            raise TypeError(f"XR camera feed {cfg.camera_name!r} does not reference a CameraCfg.")
        if not any(data_type in {"rgb", "rgba"} for data_type in camera_cfg.data_types):
            raise ValueError(f"XR camera feed {cfg.camera_name!r} camera must provide RGB or RGBA output.")
        _configure_ray_reconstruction_compatibility(camera_cfg, cfg.camera_name)
        prepared.append(cfg)
    return prepared


def _feeds_require_responsive_denoising(env_cfg: Any, cfgs: list[XrCameraFeedCfg]) -> bool:
    scene = env_cfg.scene
    return any(
        getattr(getattr(getattr(scene, cfg.camera_name), "renderer_cfg", None), "enable_dlss_ray_reconstruction", None)
        is True
        for cfg in cfgs
    )


class _XrCameraFeedSession:
    """Internal two-phase lifecycle shared by the teleoperation entry points."""

    def __init__(
        self,
        cfgs: list[XrCameraFeedCfg],
        layout_cfg: XrCameraFeedLayoutCfg | None,
        presenter: Any | None,
        *,
        requires_responsive_denoising: bool,
    ):
        self._cfgs = cfgs
        self._layout_cfg = layout_cfg
        self._presenter = presenter
        self._requires_responsive_denoising = requires_responsive_denoising
        self._env = None
        self._manager = None
        self._bound = False

    @classmethod
    def prepare(
        cls,
        env_cfg: Any,
        *,
        enabled: bool,
        camera_rendering_enabled: bool,
    ) -> _XrCameraFeedSession:
        """Prepare task-configured camera feeds before constructing the environment."""
        if type(enabled) is not bool or type(camera_rendering_enabled) is not bool:
            raise TypeError("enabled and camera_rendering_enabled must be bool values.")
        teleop_cfg = getattr(env_cfg, "isaac_teleop", None)
        if not enabled or teleop_cfg is None:
            return cls([], None, None, requires_responsive_denoising=False)

        requested = [cfg for cfg in teleop_cfg.xr_camera_feeds if cfg.enabled]
        if not camera_rendering_enabled:
            if requested:
                logger.warning("XR camera PiP is disabled because external camera rendering is disabled.")
            return cls([], None, None, requires_responsive_denoising=False)
        if not requested:
            return cls([], teleop_cfg.xr_camera_feed_layout, None, requires_responsive_denoising=False)

        _validate_layout_cfg(teleop_cfg.xr_camera_feed_layout)
        presenter = _load_kit_scene_ui_presenter()
        if presenter is None:
            return cls([], teleop_cfg.xr_camera_feed_layout, None, requires_responsive_denoising=False)
        if int(env_cfg.scene.num_envs) != 1:
            raise ValueError("XR camera PiP supports exactly one environment; set --num_envs 1 or disable PiP feeds.")
        cfgs = _prepare_camera_feed_cfgs(env_cfg, requested)
        return cls(
            cfgs,
            teleop_cfg.xr_camera_feed_layout,
            presenter,
            requires_responsive_denoising=_feeds_require_responsive_denoising(env_cfg, cfgs),
        )

    @property
    def enabled(self) -> bool:
        return bool(self._cfgs)

    @property
    def requires_responsive_denoising(self) -> bool:
        return self._requires_responsive_denoising

    def bind(self, env: Any) -> _XrCameraFeedSession:
        if self._bound:
            raise RuntimeError("XR camera feed session is already bound.")
        self._env = env
        self._bound = True
        try:
            if self.enabled:
                self._manager = _XrCameraFeedManager(env, self._cfgs, self._layout_cfg, self._presenter)
        except Exception:
            self.close()
            raise
        return self

    def refresh(self) -> None:
        if not self._bound:
            raise RuntimeError("XR camera feed session must be bound before refresh.")
        if self._manager is not None:
            # Request post-reset camera output before each render. Temporal
            # RTX cameras need several frames to replace their pre-reset
            # annotator contents, matching Isaac Lab's reset rerender path.
            for _ in range(3):
                self._manager.refresh(publish=False)
                self._env.sim.render()
            self._manager.refresh()

    def close(self) -> None:
        if self._manager is not None:
            self._manager.close()
            self._manager = None
        self._env = None
        self._bound = False

    def __enter__(self) -> _XrCameraFeedSession:
        if not self._bound:
            raise RuntimeError("Call bind(env) before entering an XR camera feed session.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _panel_size_m(cfg: XrCameraFeedCfg, image_size: tuple[int, int]) -> tuple[float, float]:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"XR camera feed image dimensions must be positive, got {image_size}.")
    return cfg.panel_width_m, cfg.panel_width_m * height / width + (0.04 if cfg.label else 0.0)


def _centered_positions(sizes: list[float], gap: float) -> list[float]:
    cursor = -0.5 * (sum(sizes) + gap * max(0, len(sizes) - 1))
    positions = []
    for size in sizes:
        positions.append(cursor + 0.5 * size)
        cursor += size + gap
    return positions


def _validate_layout_cfg(layout_cfg: XrCameraFeedLayoutCfg) -> None:
    if layout_cfg.mode not in {"manual", "horizontal", "vertical", "grid"}:
        raise ValueError(f"Unknown XR camera feed layout mode {layout_cfg.mode!r}.")
    if layout_cfg.placement not in {"viewer_start", "head_locked", "world"}:
        raise ValueError(f"Unknown XR camera feed placement {layout_cfg.placement!r}.")
    if layout_cfg.placement != "world" and (not math.isfinite(layout_cfg.distance_m) or layout_cfg.distance_m <= 0.0):
        raise ValueError("XR camera feed layout distance_m must be finite and positive.")
    if not math.isfinite(layout_cfg.panel_gap_m) or layout_cfg.panel_gap_m < 0.0:
        raise ValueError("XR camera feed layout panel_gap_m must be finite and non-negative.")
    if len(layout_cfg.center_offset_m) != 2 or not all(math.isfinite(value) for value in layout_cfg.center_offset_m):
        raise ValueError("XR camera feed layout center_offset_m must contain two finite values.")
    if type(layout_cfg.max_columns) is not int or layout_cfg.max_columns <= 0:
        raise ValueError("XR camera feed layout max_columns must be positive.")
    if layout_cfg.placement == "world":
        position = layout_cfg.world_position_m
        if position is None or len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError("XR camera feed layout world_position_m must contain three finite values.")
        orientation = layout_cfg.world_orientation_xyzw
        if len(orientation) != 4 or not all(math.isfinite(value) for value in orientation):
            raise ValueError("XR camera feed layout world_orientation_xyzw must contain four finite values.")
        if math.sqrt(sum(value * value for value in orientation)) <= 1.0e-8:
            raise ValueError("XR camera feed layout world_orientation_xyzw must be non-zero.")


def _layout_feed_cfgs(
    cfgs: list[XrCameraFeedCfg],
    image_sizes: list[tuple[int, int]],
    layout_cfg: XrCameraFeedLayoutCfg,
) -> list[XrCameraFeedCfg]:
    if len(cfgs) != len(image_sizes):
        raise ValueError("XR camera feed configs and image sizes must have the same length.")
    _validate_layout_cfg(layout_cfg)
    resolved = deepcopy(cfgs)
    if layout_cfg.mode == "manual" or not resolved:
        return resolved

    panel_sizes = [_panel_size_m(cfg, size) for cfg, size in zip(resolved, image_sizes, strict=True)]
    center_x, center_y = layout_cfg.center_offset_m
    if layout_cfg.mode == "horizontal":
        xs = _centered_positions([width for width, _ in panel_sizes], layout_cfg.panel_gap_m)
        offsets = [(center_x + x, center_y) for x in xs]
    elif layout_cfg.mode == "vertical":
        ys = _centered_positions([height for _, height in panel_sizes], layout_cfg.panel_gap_m)
        offsets = [(center_x, center_y - y) for y in ys]
    else:
        columns = min(layout_cfg.max_columns, len(resolved))
        rows = [list(range(start, min(start + columns, len(resolved)))) for start in range(0, len(resolved), columns)]
        row_heights = [max(panel_sizes[index][1] for index in row) for row in rows]
        row_ys = _centered_positions(row_heights, layout_cfg.panel_gap_m)
        offsets = [(0.0, 0.0)] * len(resolved)
        for row, row_y in zip(rows, row_ys, strict=True):
            row_xs = _centered_positions([panel_sizes[index][0] for index in row], layout_cfg.panel_gap_m)
            for index, row_x in zip(row, row_xs, strict=True):
                offsets[index] = (center_x + row_x, center_y - row_y)

    for cfg, offset in zip(resolved, offsets, strict=True):
        cfg.offset_m = offset
        cfg.distance_m = layout_cfg.distance_m
    return resolved


@dataclass(frozen=True)
class _PanelDescriptor:
    label: str | None
    width_m: float
    offset_m: tuple[float, float]
    distance_m: float
    placement: str
    world_position_m: tuple[float, float, float] | None
    world_orientation_xyzw: tuple[float, float, float, float]


def _panel_descriptor(cfg: XrCameraFeedCfg, layout_cfg: XrCameraFeedLayoutCfg) -> _PanelDescriptor:
    return _PanelDescriptor(
        label=cfg.label,
        width_m=cfg.panel_width_m,
        offset_m=tuple(cfg.offset_m),
        distance_m=cfg.distance_m,
        placement=layout_cfg.placement,
        world_position_m=None if layout_cfg.world_position_m is None else tuple(layout_cfg.world_position_m),
        world_orientation_xyzw=tuple(layout_cfg.world_orientation_xyzw),
    )


@dataclass
class _ActiveFeed:
    cfg: XrCameraFeedCfg
    camera: Camera
    image: torch.Tensor
    upload_image: torch.Tensor
    panel: Any
    next_update_time: float = 0.0


class _XrCameraFeedManager:
    """Bind camera RGBA buffers to persistent Kit SceneUI panels."""

    def __init__(
        self,
        env: Any,
        cfgs: list[XrCameraFeedCfg],
        layout_cfg: XrCameraFeedLayoutCfg | None,
        presenter: Any,
    ):
        self._env = env
        self._feeds: list[_ActiveFeed] = []
        self._presenter = presenter
        self._layout_cfg = deepcopy(layout_cfg or XrCameraFeedLayoutCfg())
        self._frame_subscription = None
        _validate_layout_cfg(self._layout_cfg)

        try:
            bound_feeds = []
            for cfg in cfgs:
                self._validate_cfg(cfg, self._layout_cfg.placement)
                camera, image = self._bind_image(cfg)
                upload_image = self._presenter.prepare_upload_image(cfg.camera_name, image)
                bound_feeds.append((camera, image, upload_image))
            image_sizes = [(int(image.shape[1]), int(image.shape[0])) for _, image, _ in bound_feeds]
            resolved_cfgs = _layout_feed_cfgs(cfgs, image_sizes, self._layout_cfg)
            for cfg, (camera, image, upload_image) in zip(resolved_cfgs, bound_feeds, strict=True):
                panel = self._presenter.create_panel(
                    _panel_descriptor(cfg, self._layout_cfg),
                    width=int(image.shape[1]),
                    height=int(image.shape[0]),
                )
                self._feeds.append(_ActiveFeed(cfg, camera, image, upload_image, panel))
            self._frame_subscription = self._presenter.subscribe_to_frame_updates(self._on_frame)
        except Exception:
            self.close()
            raise

    def _on_frame(self, _event: Any) -> None:
        self.update()

    @staticmethod
    def _validate_cfg(cfg: XrCameraFeedCfg, placement: str) -> None:
        if not math.isfinite(cfg.panel_width_m) or cfg.panel_width_m <= 0.0:
            raise ValueError(f"panel_width_m for XR camera feed {cfg.camera_name!r} must be positive.")
        if len(cfg.offset_m) != 2 or not all(math.isfinite(value) for value in cfg.offset_m):
            raise ValueError(f"offset_m for XR camera feed {cfg.camera_name!r} must contain two finite values.")
        if placement != "world" and (not math.isfinite(cfg.distance_m) or cfg.distance_m <= 0.0):
            raise ValueError(f"distance_m for XR camera feed {cfg.camera_name!r} must be positive.")
        if not math.isfinite(cfg.max_update_hz) or cfg.max_update_hz < 0.0:
            raise ValueError(f"max_update_hz for XR camera feed {cfg.camera_name!r} must be finite and non-negative.")

    def _bind_image(self, cfg: XrCameraFeedCfg) -> tuple[Camera, torch.Tensor]:
        sensors = getattr(getattr(self._env, "scene", None), "sensors", {})
        if cfg.camera_name not in sensors:
            raise ValueError(
                f"XR camera feed {cfg.camera_name!r} is not present in the interactive scene. "
                f"Available sensors: {sorted(sensors.keys())}."
            )
        camera = sensors[cfg.camera_name]
        if not isinstance(camera, _camera_type()):
            raise TypeError(f"XR camera feed {cfg.camera_name!r} did not resolve to an Isaac Lab Camera.")
        return camera, self._image_from_output(cfg, camera.data.output)

    def _image_from_output(self, cfg: XrCameraFeedCfg, output: Any) -> torch.Tensor:
        if output is None or "rgba" not in output:
            available = [] if output is None else sorted(output.keys())
            raise ValueError(f"Camera {cfg.camera_name!r} has no RGBA buffer. Available outputs: {available}.")
        batch = output["rgba"].torch
        if batch.ndim != 4 or int(batch.shape[-1]) != 4:
            raise ValueError(f"Camera {cfg.camera_name!r} RGBA output must have shape (N, H, W, 4).")
        image = batch[0]
        if image.dtype != torch.uint8:
            raise TypeError(f"Camera {cfg.camera_name!r} RGBA buffer must be uint8, got {image.dtype}.")
        return image

    @staticmethod
    def _same_allocation(first: torch.Tensor, second: torch.Tensor) -> bool:
        return (
            tuple(first.shape) == tuple(second.shape)
            and first.device == second.device
            and first.data_ptr() == second.data_ptr()
        )

    def _rebind_feed(self, feed: _ActiveFeed, camera: Camera, image: torch.Tensor) -> None:
        if self._same_allocation(feed.image, image):
            feed.camera = camera
            return
        old_shape = tuple(feed.image.shape)
        feed.upload_image = self._presenter.prepare_upload_image(
            feed.cfg.camera_name,
            image,
            previous_source=feed.image,
            previous_upload=feed.upload_image,
        )
        if tuple(image.shape) != old_shape:
            replacement = self._presenter.create_panel(
                _panel_descriptor(feed.cfg, self._layout_cfg),
                width=int(image.shape[1]),
                height=int(image.shape[0]),
            )
            old_panel = feed.panel
            feed.panel = replacement
            old_panel.close()
        feed.camera = camera
        feed.image = image

    def update(self) -> None:
        now = time.monotonic()
        for feed in self._feeds:
            if now < feed.next_update_time:
                continue
            image = self._image_from_output(feed.cfg, feed.camera.data.output)
            self._rebind_feed(feed, feed.camera, image)
            self._publish_feed(feed)
            period = 0.0 if feed.cfg.max_update_hz == 0.0 else 1.0 / feed.cfg.max_update_hz
            feed.next_update_time = now + period

    def _publish_feed(self, feed: _ActiveFeed) -> None:
        self._presenter.stage_upload_image(feed.image, feed.upload_image)
        feed.panel.upload(feed.upload_image)

    def refresh(self, *, publish: bool = True) -> None:
        for feed in self._feeds:
            feed.camera.update(0.0, force_recompute=True)
            camera, image = self._bind_image(feed.cfg)
            self._rebind_feed(feed, camera, image)
            if publish:
                self._publish_feed(feed)
            feed.next_update_time = 0.0

    def close(self) -> None:
        if self._frame_subscription is not None:
            self._frame_subscription.close()
            self._frame_subscription = None
        for feed in reversed(self._feeds):
            try:
                feed.panel.close()
            except Exception:
                logger.exception("Failed to close XR camera feed %r.", feed.cfg.camera_name)
        self._feeds.clear()
