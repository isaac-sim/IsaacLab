# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viser-based visualizer using Newton's ViewerViser."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from newton.viewer import ViewerViser

from .viser_visualizer_cfg import ViserVisualizerCfg
from .visualizer import Visualizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.sim.scene_data_providers import SceneDataProvider


def _disable_viser_runtime_client_rebuild_if_bundled() -> None:
    """Skip viser's runtime frontend rebuild when a bundled build is present.

    Newer viser versions may try to rebuild the client if source timestamps are newer
    than the packaged build directory, which requires ``nodeenv`` at runtime.
    For Isaac Lab usage, we prefer the prebuilt static assets shipped with the package.
    """

    try:
        import viser
        import viser._client_autobuild as client_autobuild
    except Exception:
        return

    client_root = Path(viser.__file__).resolve().parent / "client"
    has_bundled_build = (client_root / "build" / "index.html").exists()
    if not has_bundled_build:
        return

    client_autobuild.ensure_client_is_built = lambda: None


class NewtonViewerViser(ViewerViser):
    """Isaac Lab wrapper for Newton's ViewerViser."""

    def __init__(
        self,
        port: int = 8080,
        label: str | None = None,
        verbose: bool = True,
        share: bool = False,
        record_to_viser: str | None = None,
        metadata: dict | None = None,
    ):
        _disable_viser_runtime_client_rebuild_if_bundled()
        super().__init__(
            port=port,
            label=label,
            verbose=verbose,
            share=share,
            record_to_viser=record_to_viser,
        )
        self._metadata = metadata or {}


class ViserVisualizer(Visualizer):
    """Viser web-based visualizer backed by Newton's ViewerViser."""

    def __init__(self, cfg: ViserVisualizerCfg):
        super().__init__(cfg)
        self.cfg: ViserVisualizerCfg = cfg
        self._viewer: NewtonViewerViser | None = None
        self._model: Any | None = None
        self._state = None
        self._is_initialized = False
        self._sim_time = 0.0
        self._scene_data_provider = None
        self._active_record_path: str | None = None
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._pending_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        if self._is_initialized:
            logger.debug("[ViserVisualizer] initialize() called while already initialized.")
            return
        if scene_data_provider is None:
            raise RuntimeError("Viser visualizer requires a scene_data_provider.")

        self._scene_data_provider = scene_data_provider
        metadata = scene_data_provider.get_metadata()
        self._env_ids = self._compute_visualized_env_ids()
        if self._env_ids:
            get_filtered_model = getattr(scene_data_provider, "get_newton_model_for_env_ids", None)
            if callable(get_filtered_model):
                self._model = get_filtered_model(self._env_ids)
            else:
                self._model = scene_data_provider.get_newton_model()
        else:
            self._model = scene_data_provider.get_newton_model()
        self._state = scene_data_provider.get_newton_state(self._env_ids)

        try:
            self._active_record_path = self.cfg.record_to_viser
            self._create_viewer(record_to_viser=self.cfg.record_to_viser, metadata=metadata)
            logger.info(
                "[ViserVisualizer] initialized | camera_pos=%s camera_target=%s",
                self.cfg.camera_position,
                self.cfg.camera_target,
            )
            self._is_initialized = True
        except Exception as exc:
            logger.error(f"[ViserVisualizer] Failed to initialize viewer: {exc}")
            raise

    def step(self, dt: float, state: Any | None = None) -> None:
        if not self._is_initialized or self._viewer is None or self._scene_data_provider is None:
            return

        if self.cfg.camera_source == "usd_path":
            self._update_camera_from_usd_path()
        self._apply_pending_camera_pose()

        self._state = self._scene_data_provider.get_newton_state(self._env_ids)
        self._sim_time += dt

        self._viewer.begin_frame(self._sim_time)
        self._viewer.log_state(self._state)
        self._viewer.end_frame()

    def close(self) -> None:
        if not self._is_initialized:
            return

        try:
            self._close_viewer(finalize_viser=bool(self.cfg.record_to_viser))
        except Exception as exc:
            logger.warning(f"[ViserVisualizer] Error during close: {exc}")

        self._viewer = None
        self._is_initialized = False
        self._is_closed = True
        self._active_record_path = None
        self._pending_camera_pose = None

    def is_running(self) -> bool:
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def is_training_paused(self) -> bool:
        return False

    def supports_markers(self) -> bool:
        return False

    def supports_live_plots(self) -> bool:
        return False

    def _create_viewer(self, record_to_viser: str | None, metadata: dict | None = None) -> None:
        if self._model is None:
            raise RuntimeError("Viser visualizer requires a Newton model.")

        self._viewer = NewtonViewerViser(
            port=self.cfg.port,
            label=self.cfg.label,
            verbose=self.cfg.verbose,
            share=self.cfg.share,
            record_to_viser=record_to_viser,
            metadata=metadata or {},
        )
        self._viewer.set_model(self._model)
        self._set_viser_camera_view(self._resolve_initial_camera_pose())
        self._sim_time = 0.0

    def _close_viewer(self, finalize_viser: bool = False) -> None:
        if self._viewer is None:
            return
        self._viewer.close()
        if finalize_viser and self._active_record_path:
            if os.path.exists(self._active_record_path):
                size = os.path.getsize(self._active_record_path)
                logger.info(
                    "[ViserVisualizer] Recording saved: %s (%s bytes)",
                    self._active_record_path,
                    size,
                )
            else:
                logger.warning("[ViserVisualizer] Recording file not found: %s", self._active_record_path)
        self._viewer = None

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if self.cfg.camera_source == "usd_path":
            pose = self._resolve_camera_pose_from_usd_path(self.cfg.camera_usd_path)
            if pose is not None:
                return pose
            logger.warning(
                "[ViserVisualizer] camera_usd_path '%s' not found; using configured camera.",
                self.cfg.camera_usd_path,
            )
        return self.cfg.camera_position, self.cfg.camera_target

    def _try_apply_viser_camera_view(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> bool:
        if self._viewer is None:
            return False

        server = getattr(self._viewer, "_server", None)
        get_clients = getattr(server, "get_clients", None) if server is not None else None
        if not callable(get_clients):
            return False

        try:
            clients = get_clients()
        except Exception:
            return False

        if isinstance(clients, dict):
            client_iterable = clients.values()
        else:
            client_iterable = clients

        cam_pos, cam_target = pose
        applied = False
        for client in client_iterable:
            camera = getattr(client, "camera", None)
            if camera is None:
                continue
            try:
                if hasattr(camera, "position"):
                    camera.position = cam_pos
                    applied = True
                if hasattr(camera, "look_at"):
                    camera.look_at = cam_target
                    applied = True
            except Exception:
                continue
        return applied

    def _set_viser_camera_view(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        if self._try_apply_viser_camera_view(pose):
            self._last_camera_pose = pose
            self._pending_camera_pose = None
        else:
            self._pending_camera_pose = pose

    def _apply_pending_camera_pose(self) -> None:
        if self._pending_camera_pose is None:
            return
        if self._try_apply_viser_camera_view(self._pending_camera_pose):
            self._last_camera_pose = self._pending_camera_pose
            self._pending_camera_pose = None

    def _update_camera_from_usd_path(self) -> None:
        pose = self._resolve_camera_pose_from_usd_path(self.cfg.camera_usd_path)
        if pose is None:
            return
        if self._last_camera_pose == pose or self._pending_camera_pose == pose:
            return
        self._set_viser_camera_view(pose)
