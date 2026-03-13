# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viser-based visualizer using Newton's ViewerViser."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from newton.viewer import ViewerViser

from isaaclab.visualizers.base_visualizer import BaseVisualizer

from .viser_visualizer_cfg import ViserVisualizerCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.physics import BaseSceneDataProvider


def _disable_viser_runtime_client_rebuild_if_bundled() -> None:
    """Skip viser's runtime frontend rebuild when a bundled build is present."""
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


@contextlib.contextmanager
def _suppress_viser_startup_logs(enabled: bool):
    """Temporarily quiet noisy viser/websockets startup output."""
    if not enabled:
        yield
        return

    websockets_logger = logging.getLogger("websockets.server")
    previous_level = websockets_logger.level
    websockets_logger.setLevel(logging.WARNING)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        websockets_logger.setLevel(previous_level)


def _open_viser_web_viewer(port: int) -> None:
    """Open the local viser web UI in a browser."""
    url = _viser_web_viewer_url(port)
    try:
        if not webbrowser.open_new_tab(url):
            logger.info("[ViserVisualizer] Could not auto-open browser tab. Open manually: %s", url)
    except Exception:
        logger.info("[ViserVisualizer] Could not auto-open browser tab. Open manually: %s", url)


def _viser_web_viewer_url(port: int) -> str:
    """Return local viser web UI URL."""
    return f"http://localhost:{int(port)}"


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
        """Initialize Newton-backed viser viewer wrapper.

        Args:
            port: HTTP port for viser server.
            label: Optional viewer label.
            verbose: Whether to keep verbose startup output enabled.
            share: Whether to enable sharing/tunneling.
            record_to_viser: Optional recording destination.
            metadata: Optional metadata attached to the viewer.
        """
        _disable_viser_runtime_client_rebuild_if_bundled()
        super().__init__(
            port=port,
            label=label,
            verbose=verbose,
            share=share,
            record_to_viser=record_to_viser,
        )
        self._metadata = metadata or {}


class ViserVisualizer(BaseVisualizer):
    """Viser web-based visualizer backed by Newton's ViewerViser."""

    def __init__(self, cfg: ViserVisualizerCfg):
        """Initialize Viser visualizer state.

        Args:
            cfg: Viser visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: ViserVisualizerCfg = cfg
        self._viewer: NewtonViewerViser | None = None
        self._model: Any | None = None
        self._state = None
        self._sim_time = 0.0
        self._scene_data_provider = None
        self._active_record_path: str | None = None
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._pending_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def initialize(self, scene_data_provider: BaseSceneDataProvider) -> None:
        """Initialize viewer resources and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used to fetch model/state data.
        """
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
            self._model = (
                get_filtered_model(self._env_ids)
                if callable(get_filtered_model)
                else scene_data_provider.get_newton_model()
            )
        else:
            self._model = scene_data_provider.get_newton_model()
        self._state = scene_data_provider.get_newton_state(self._env_ids)

        self._active_record_path = self.cfg.record_to_viser
        self._create_viewer(record_to_viser=self.cfg.record_to_viser, metadata=metadata)
        num_visualized_envs = len(self._env_ids) if self._env_ids is not None else int(metadata.get("num_envs", 0))
        viewer_url = _viser_web_viewer_url(self.cfg.port)
        self._log_initialization_table(
            logger=logger,
            title="ViserVisualizer Configuration",
            rows=[
                ("camera_position", self.cfg.camera_position),
                ("camera_target", self.cfg.camera_target),
                ("camera_source", self.cfg.camera_source),
                ("num_visualized_envs", num_visualized_envs),
                ("port", self.cfg.port),
                ("viewer_url", viewer_url),
                ("record_to_viser", self.cfg.record_to_viser or "<none>"),
            ],
        )
        self._is_initialized = True

    def step(self, dt: float) -> None:
        """Advance visualization by one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
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
        """Close viewer resources and finalize optional recording."""
        if not self._is_initialized:
            return
        try:
            self._close_viewer(finalize_viser=bool(self.cfg.record_to_viser))
        except Exception as exc:
            logger.warning("[ViserVisualizer] Error during close: %s", exc)

        self._viewer = None
        self._is_initialized = False
        self._is_closed = True
        self._active_record_path = None
        self._pending_camera_pose = None

    def is_running(self) -> bool:
        """Return whether the visualizer should continue stepping.

        Returns:
            ``True`` while the visualizer is active, otherwise ``False``.
        """
        if not self._is_initialized or self._is_closed:
            return False
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def is_training_paused(self) -> bool:
        """Return whether training is paused.

        Viser backend does not currently expose a training pause control.
        """
        return False

    def supports_markers(self) -> bool:
        """Viser backend currently does not expose Isaac Lab marker primitives."""
        return False

    def supports_live_plots(self) -> bool:
        """Viser backend currently does not expose Isaac Lab live-plot widgets."""
        return False

    def _create_viewer(self, record_to_viser: str | None, metadata: dict | None = None) -> None:
        """Create Newton-backed Viser viewer and apply initial camera.

        Args:
            record_to_viser: Optional output path for viser recording.
            metadata: Optional metadata passed to viewer.
        """
        if self._model is None:
            raise RuntimeError("Viser visualizer requires a Newton model.")

        with _suppress_viser_startup_logs(enabled=not self.cfg.verbose):
            self._viewer = NewtonViewerViser(
                port=self.cfg.port,
                label=self.cfg.label,
                verbose=self.cfg.verbose,
                share=self.cfg.share,
                record_to_viser=record_to_viser,
                metadata=metadata or {},
            )
        max_worlds = self.cfg.max_worlds
        self._viewer.set_model(self._model, max_worlds=max_worlds)
        # Preserve simulation world positions (env_spacing) rather than adding viewer-side offsets.
        self._viewer.set_world_offsets((0.0, 0.0, 0.0))
        if self.cfg.open_browser:
            _open_viser_web_viewer(self.cfg.port)
        self._set_viser_camera_view(self._resolve_initial_camera_pose())
        self._sim_time = 0.0

    def _close_viewer(self, finalize_viser: bool = False) -> None:
        """Close viewer and log recording output when requested."""
        if self._viewer is None:
            return
        self._viewer.close()
        if finalize_viser and self._active_record_path:
            if os.path.exists(self._active_record_path):
                size = os.path.getsize(self._active_record_path)
                logger.info("[ViserVisualizer] Recording saved: %s (%s bytes)", self._active_record_path, size)
            else:
                logger.warning("[ViserVisualizer] Recording file not found: %s", self._active_record_path)
        self._viewer = None

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Resolve initial camera pose from config or USD camera path."""
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
        """Try applying camera pose to active viser clients.

        Returns:
            ``True`` if at least one client camera was updated, otherwise ``False``.
        """
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

        client_iterable = clients.values() if isinstance(clients, dict) else clients
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
        """Apply or defer camera pose update depending on client readiness."""
        if self._try_apply_viser_camera_view(pose):
            self._last_camera_pose = pose
            self._pending_camera_pose = None
        else:
            self._pending_camera_pose = pose

    def _apply_pending_camera_pose(self) -> None:
        """Apply deferred camera pose once client cameras are available."""
        if self._pending_camera_pose is None:
            return
        if self._try_apply_viser_camera_view(self._pending_camera_pose):
            self._last_camera_pose = self._pending_camera_pose
            self._pending_camera_pose = None

    def _update_camera_from_usd_path(self) -> None:
        """Refresh camera pose from configured USD camera path when it changes."""
        pose = self._resolve_camera_pose_from_usd_path(self.cfg.camera_usd_path)
        if pose is None:
            return
        if self._last_camera_pose == pose or self._pending_camera_pose == pose:
            return
        self._set_viser_camera_view(pose)
