# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rerun-based visualizer using rerun-sdk."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING, Any

import rerun as rr
import rerun.blueprint as rrb
from newton.viewer import ViewerRerun

from .rerun_visualizer_cfg import RerunVisualizerCfg
from .visualizer import Visualizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.sim.scene_data_providers import SceneDataProvider


class NewtonViewerRerun(ViewerRerun):
    """Isaac Lab wrapper for Newton's ViewerRerun."""

    def __init__(
        self,
        app_id: str | None = None,
        address: str | None = None,
        web_port: int | None = None,
        grpc_port: int | None = None,
        keep_historical_data: bool = False,
        keep_scalar_history: bool = True,
        record_to_rrd: str | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(
            app_id=app_id,
            address=address,
            web_port=web_port,
            grpc_port=grpc_port or 9876,
            serve_web_viewer=True,
            keep_historical_data=keep_historical_data,
            keep_scalar_history=keep_scalar_history,
            record_to_rrd=record_to_rrd,
        )

        self._metadata = metadata or {}
        self._log_metadata()

    def _log_metadata(self) -> None:
        metadata_text = "# Isaac Lab Scene Metadata\n\n"
        physics_backend = self._metadata.get("physics_backend", "unknown")
        metadata_text += f"**Physics Backend:** {physics_backend}\n"
        num_envs = self._metadata.get("num_envs", 0)
        metadata_text += f"**Total Environments:** {num_envs}\n"

        for key, value in self._metadata.items():
            if key not in ["physics_backend", "num_envs"]:
                metadata_text += f"**{key}:** {value}\n"

        rr.log("metadata", rr.TextDocument(metadata_text, media_type=rr.MediaType.MARKDOWN))


class RerunVisualizer(Visualizer):
    """Rerun web-based visualizer with time scrubbing and data inspection."""

    def __init__(self, cfg: RerunVisualizerCfg):
        super().__init__(cfg)
        self.cfg: RerunVisualizerCfg = cfg
        self._viewer: NewtonViewerRerun | None = None
        self._model = None
        self._state = None
        self._is_initialized = False
        self._sim_time = 0.0
        self._scene_data_provider = None
        self._rerun_server_process = None
        self._rerun_address: str | None = None
        self._active_record_path: str | None = None
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        if self._is_initialized:
            logger.debug("[RerunVisualizer] initialize() called while already initialized.")
            return
        if scene_data_provider is None:
            raise RuntimeError("Rerun visualizer requires a scene_data_provider.")

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
            self._setup_rerun_server()
            self._active_record_path = self.cfg.record_to_rrd
            self._create_viewer(record_to_rrd=self.cfg.record_to_rrd, metadata=metadata)
            logger.info(
                "[RerunVisualizer] initialized | camera_pos=%s camera_target=%s",
                self.cfg.camera_position,
                self.cfg.camera_target,
            )
            self._is_initialized = True
        except Exception as exc:
            logger.error(f"[RerunVisualizer] Failed to initialize viewer: {exc}")
            raise

    def step(self, dt: float, state: Any | None = None) -> None:
        if not self._is_initialized or self._viewer is None or self._scene_data_provider is None:
            return

        if self.cfg.camera_source == "usd_path":
            self._update_camera_from_usd_path()

        self._state = self._scene_data_provider.get_newton_state(self._env_ids)
        self._sim_time += dt

        self._viewer.begin_frame(self._sim_time)
        self._viewer.log_state(self._state)
        self._viewer.end_frame()

    def close(self) -> None:
        if not self._is_initialized:
            return

        try:
            self._close_viewer(finalize_rrd=bool(self.cfg.record_to_rrd))
        except Exception as exc:
            logger.warning(f"[RerunVisualizer] Error during close: {exc}")

        self._viewer = None
        self._is_initialized = False
        self._is_closed = True
        self._active_record_path = None
        if self._rerun_server_process is not None:
            with contextlib.suppress(Exception):
                self._rerun_server_process.terminate()
            self._rerun_server_process = None

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

    def _setup_rerun_server(self) -> None:
        if not self.cfg.bind_address or self._rerun_server_process is not None:
            return
        import shutil
        import subprocess

        rerun_bin = shutil.which("rerun")
        if rerun_bin is None:
            logger.warning("[RerunVisualizer] 'rerun' binary not found in PATH. Skipping external bind.")
            return

        cmd = [
            rerun_bin,
            "--serve-web",
            "--bind",
            self.cfg.bind_address,
            "--port",
            str(self.cfg.grpc_port),
            "--web-viewer-port",
            str(self.cfg.web_port),
        ]
        if self.cfg.open_browser:
            cmd.append("--web-viewer")
        self._rerun_server_process = subprocess.Popen(cmd)
        logger.info(
            "[RerunVisualizer] Server bind %s:%s, web %s",
            self.cfg.bind_address,
            self.cfg.grpc_port,
            self.cfg.web_port,
        )
        self._rerun_address = f"rerun+http://127.0.0.1:{self.cfg.grpc_port}/proxy"

    def _create_viewer(self, record_to_rrd: str | None, metadata: dict | None = None) -> None:
        self._viewer = NewtonViewerRerun(
            app_id=self.cfg.app_id,
            address=self._rerun_address,
            web_port=self.cfg.web_port,
            grpc_port=self.cfg.grpc_port,
            keep_historical_data=self.cfg.keep_historical_data,
            keep_scalar_history=self.cfg.keep_scalar_history,
            record_to_rrd=record_to_rrd,
            metadata=metadata or {},
        )
        self._viewer.set_model(self._model)
        self._set_rerun_camera_view(self._resolve_initial_camera_pose())

        self._sim_time = 0.0

    def _close_viewer(self, finalize_rrd: bool = False) -> None:
        if self._viewer is None:
            return
        self._viewer.close()
        if finalize_rrd and self._active_record_path:
            if os.path.exists(self._active_record_path):
                size = os.path.getsize(self._active_record_path)
                logger.info(
                    "[RerunVisualizer] Recording saved: %s (%s bytes)",
                    self._active_record_path,
                    size,
                )
            else:
                logger.warning("[RerunVisualizer] Recording file not found: %s", self._active_record_path)
        self._viewer = None

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if self.cfg.camera_source == "usd_path":
            pose = self._resolve_camera_pose_from_usd_path(self.cfg.camera_usd_path)
            if pose is not None:
                return pose
            logger.warning(
                "[RerunVisualizer] camera_usd_path '%s' not found; using configured camera.",
                self.cfg.camera_usd_path,
            )
        return self.cfg.camera_position, self.cfg.camera_target

    def _set_rerun_camera_view(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        cam_pos, cam_target = pose
        try:
            blueprint = rrb.Blueprint(
                rrb.Spatial3DView(
                    name="3D View",
                    origin="/",
                    eye_controls=rrb.EyeControls3D(
                        position=cam_pos,
                        look_target=cam_target,
                    ),
                ),
                collapse_panels=True,
            )
            rr.send_blueprint(blueprint)
            self._last_camera_pose = (cam_pos, cam_target)
        except Exception as exc:
            logger.warning(f"[RerunVisualizer] Could not set camera view: {exc}")

    def _update_camera_from_usd_path(self) -> None:
        pose = self._resolve_camera_pose_from_usd_path(self.cfg.camera_usd_path)
        if pose is None:
            return
        if self._last_camera_pose == pose:
            return
        self._set_rerun_camera_view(pose)
