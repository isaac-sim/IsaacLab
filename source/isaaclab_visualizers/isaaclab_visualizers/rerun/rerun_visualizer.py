# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rerun visualizer implementation for Isaac Lab."""

from __future__ import annotations

import atexit
import logging
import socket
import webbrowser
from typing import TYPE_CHECKING
from urllib.parse import quote

import rerun as rr
import rerun.blueprint as rrb
from newton.viewer import ViewerRerun

from isaaclab.visualizers.base_visualizer import BaseVisualizer

from .rerun_visualizer_cfg import RerunVisualizerCfg

if TYPE_CHECKING:
    from isaaclab.physics import BaseSceneDataProvider

logger = logging.getLogger(__name__)


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
            return True
        except OSError:
            return False


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, int(port))) == 0


def _normalize_host(addr: str) -> str:
    if addr in ("0.0.0.0", "127.0.0.1", "localhost"):
        return "127.0.0.1"
    return addr


def _stop_managed_rerun_server() -> None:
    return


def _ensure_rerun_server(app_id: str, bind_address: str, grpc_port: int, web_port: int) -> tuple[str, bool]:
    """Resolve rerun endpoint and whether viewer should start web/grpc server."""
    del app_id
    connect_host = _normalize_host(bind_address)
    expected_uri = f"rerun+http://{connect_host}:{int(grpc_port)}/proxy"

    if _is_port_open(grpc_port, host=connect_host):
        # Reuse existing endpoint; do not create a new server here.
        return expected_uri, False

    if not _is_port_free(web_port, host=connect_host):
        raise RuntimeError(f"Rerun web port {web_port} is in use. Free the port or choose a different `web_port`.")

    # No existing gRPC server: NewtonViewerRerun should start and own it.
    return expected_uri, True


def _open_rerun_web_viewer(host: str, web_port: int, connect_to: str) -> None:
    url = f"http://{host}:{int(web_port)}/?url={quote(connect_to, safe='')}"
    try:
        if not webbrowser.open_new_tab(url):
            logger.info("[RerunVisualizer] Could not auto-open browser tab. Open manually: %s", url)
    except Exception:
        logger.info("[RerunVisualizer] Could not auto-open browser tab. Open manually: %s", url)


class NewtonViewerRerun(ViewerRerun):
    """Wrapper around Newton's ViewerRerun with rendering pause controls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._paused_rendering = False

    def is_rendering_paused(self) -> bool:
        return self._paused_rendering

    def _render_ui(self):
        super()._render_ui()

        if not self._has_imgui:
            return

        imgui = self._imgui
        if not imgui:
            return

        if imgui.collapsing_header("IsaacLab Controls"):
            if imgui.button("Pause Rendering" if not self._paused_rendering else "Resume Rendering"):
                self._paused_rendering = not self._paused_rendering


class RerunVisualizer(BaseVisualizer):
    """Rerun visualizer for Isaac Lab."""

    def __init__(self, cfg: RerunVisualizerCfg):
        super().__init__(cfg)
        self.cfg: RerunVisualizerCfg = cfg
        self._viewer: NewtonViewerRerun | None = None
        self._sim_time = 0.0
        self._step_counter = 0
        self._model = None
        self._state = None
        self._scene_data_provider = None
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def initialize(self, scene_data_provider: BaseSceneDataProvider) -> None:
        if self._is_initialized:
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

        grpc_port = int(self.cfg.grpc_port)
        web_port = int(self.cfg.web_port)
        bind_address = self.cfg.bind_address or "0.0.0.0"
        rerun_address, start_server_in_viewer = _ensure_rerun_server(
            app_id=self.cfg.app_id,
            bind_address=bind_address,
            grpc_port=grpc_port,
            web_port=web_port,
        )
        if not start_server_in_viewer:
            logger.info("[RerunVisualizer] Reusing existing rerun server at %s.", rerun_address)

        viewer_address = None if start_server_in_viewer else rerun_address
        self._viewer = NewtonViewerRerun(
            app_id=self.cfg.app_id,
            address=viewer_address,
            serve_web_viewer=start_server_in_viewer,
            web_port=web_port,
            grpc_port=grpc_port,
            keep_historical_data=self.cfg.keep_historical_data,
            keep_scalar_history=self.cfg.keep_scalar_history,
            record_to_rrd=self.cfg.record_to_rrd,
        )
        if start_server_in_viewer:
            rerun_address = getattr(self._viewer, "_grpc_server_uri", rerun_address)
        if self.cfg.open_browser and not start_server_in_viewer:
            _open_rerun_web_viewer(_normalize_host(bind_address), web_port, rerun_address)
        self._viewer.set_model(self._model)
        self._apply_camera_pose(self._resolve_initial_camera_pose())
        self._viewer.up_axis = 2
        self._viewer.scaling = 1.0
        self._viewer._paused = False

        num_visualized_envs = len(self._env_ids) if self._env_ids is not None else int(metadata.get("num_envs", 0))
        self._log_initialization_table(
            logger=logger,
            title="RerunVisualizer Configuration",
            rows=[
                ("camera_position", self.cfg.camera_position),
                ("camera_target", self.cfg.camera_target),
                ("camera_source", self.cfg.camera_source),
                ("num_visualized_envs", num_visualized_envs),
                ("endpoint", f"http://{_normalize_host(bind_address)}:{web_port}"),
                ("bind_address", bind_address),
                ("grpc_port", grpc_port),
                ("web_port", web_port),
                ("open_browser", self.cfg.open_browser),
                ("record_to_rrd", self.cfg.record_to_rrd or "<none>"),
            ],
        )

        self._is_initialized = True
        atexit.register(self.close)

    def step(self, dt: float) -> None:
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return

        self._sim_time += dt
        self._step_counter += 1

        if self.cfg.camera_source == "usd_path":
            self._update_camera_from_usd_path()

        self._state = self._scene_data_provider.get_newton_state(self._env_ids)

        if not self._viewer.is_paused():
            self._viewer.begin_frame(self._sim_time)
            if self._state is not None:
                body_q = getattr(self._state, "body_q", None)
                if hasattr(body_q, "shape") and body_q.shape[0] == 0:
                    self._viewer.end_frame()
                    return
                self._viewer.log_state(self._state)
            self._viewer.end_frame()

    def close(self) -> None:
        if self._is_closed:
            return

        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception as exc:
                logger.warning("[RerunVisualizer] Failed while closing viewer: %s", exc)
            finally:
                self._viewer = None

        try:
            rr.disconnect()
        except Exception as exc:
            logger.warning("[RerunVisualizer] Failed while disconnecting rerun: %s", exc)
        _stop_managed_rerun_server()
        self._is_closed = True

    def is_running(self) -> bool:
        if not self._is_initialized or self._is_closed:
            return False
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if self.cfg.camera_source == "usd_path":
            pose = self._resolve_camera_pose_from_usd_path(self.cfg.camera_usd_path)
            if pose is not None:
                return pose
        return self.cfg.camera_position, self.cfg.camera_target

    def _apply_camera_pose(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        if self._viewer is None:
            return
        cam_pos, cam_target = pose
        rr.send_blueprint(
            rrb.Blueprint(
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
        )
        self._last_camera_pose = (cam_pos, cam_target)

    def _update_camera_from_usd_path(self) -> None:
        pose = self._resolve_camera_pose_from_usd_path(self.cfg.camera_usd_path)
        if pose is None:
            return
        if self._last_camera_pose == pose:
            return
        self._apply_camera_pose(pose)

    def supports_markers(self) -> bool:
        return False

    def supports_live_plots(self) -> bool:
        return False

    def is_training_paused(self) -> bool:
        return False

    def is_rendering_paused(self) -> bool:
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_rendering_paused()
