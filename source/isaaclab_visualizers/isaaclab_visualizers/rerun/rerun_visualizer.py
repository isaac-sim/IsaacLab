# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rerun visualizer implementation for Isaac Lab."""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import re
import shutil
import socket
import subprocess
import time
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
_RERUN_SERVER_PROCESS: subprocess.Popen | None = None


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


def _listening_rerun_pid(port: int) -> int | None:
    """Return rerun pid listening on port, if any."""
    try:
        proc = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None
        pattern = re.compile(rf":{int(port)}\b.*users:\(\(\"rerun\",pid=(\d+),")
        for line in proc.stdout.splitlines():
            match = pattern.search(line)
            if match:
                return int(match.group(1))
    except Exception:
        return None
    return None


def _terminate_pid(pid: int, timeout_s: float = 2.0) -> bool:
    try:
        os.kill(pid, 15)  # SIGTERM
    except OSError:
        return True
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not os.path.exists(f"/proc/{pid}"):
            return True
        time.sleep(0.05)
    with contextlib.suppress(OSError):
        os.kill(pid, 9)  # SIGKILL
    time.sleep(0.05)
    return not os.path.exists(f"/proc/{pid}")


def _stop_managed_rerun_server() -> None:
    global _RERUN_SERVER_PROCESS
    if _RERUN_SERVER_PROCESS is None:
        return
    with contextlib.suppress(OSError):
        _RERUN_SERVER_PROCESS.terminate()
    _RERUN_SERVER_PROCESS = None


def _ensure_rerun_server(
    bind_address: str, grpc_port: int, web_port: int, auto_kill_stale_rerun_process: bool
) -> tuple[str, bool]:
    """Ensure rerun server exists; return (grpc_uri, process_owned_by_this_instance)."""
    global _RERUN_SERVER_PROCESS
    connect_host = _normalize_host(bind_address)
    grpc_uri = f"rerun+http://{connect_host}:{int(grpc_port)}/proxy"

    if _is_port_open(grpc_port, host=connect_host):
        return grpc_uri, False

    if not _is_port_free(web_port, host=connect_host):
        pid = _listening_rerun_pid(web_port)
        if pid is not None and auto_kill_stale_rerun_process:
            logger.info("[RerunVisualizer] Terminating stale rerun process on web port %s (pid=%s).", web_port, pid)
            _terminate_pid(pid)
            time.sleep(0.1)
        if not _is_port_free(web_port, host=connect_host):
            if pid is not None and not auto_kill_stale_rerun_process:
                raise RuntimeError(
                    f"Rerun web port {web_port} is in use by rerun pid={pid}; "
                    "set auto_kill_stale_rerun_process=True to allow cleanup."
                )
            raise RuntimeError(f"Rerun web port {web_port} is in use and not owned by a detectable rerun process.")

    rerun_bin = shutil.which("rerun")
    if rerun_bin is None:
        raise RuntimeError("'rerun' binary not found in PATH.")

    cmd = [
        rerun_bin,
        "--serve-web",
        "--bind",
        bind_address,
        "--port",
        str(int(grpc_port)),
        "--web-viewer-port",
        str(int(web_port)),
        "--memory-limit",
        "25%",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _RERUN_SERVER_PROCESS = proc
    deadline = time.time() + 6.0
    while time.time() < deadline:
        if proc.poll() is not None:
            stderr_txt = ""
            stdout_txt = ""
            with contextlib.suppress(Exception):
                stdout_txt, stderr_txt = proc.communicate(timeout=0.2)
            _RERUN_SERVER_PROCESS = None
            detail = "\n".join(
                part
                for part in [
                    f"stderr:\n{stderr_txt.strip()}" if stderr_txt.strip() else "",
                    f"stdout:\n{stdout_txt.strip()}" if stdout_txt.strip() else "",
                ]
                if part
            ).strip()
            raise RuntimeError(detail or "rerun server exited before opening gRPC port.")
        if _is_port_open(grpc_port, host=connect_host):
            return grpc_uri, True
        time.sleep(0.1)

    _stop_managed_rerun_server()
    raise RuntimeError(f"Timed out waiting for rerun gRPC port {connect_host}:{grpc_port}.")


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
        rerun_address, owns_server = _ensure_rerun_server(
            bind_address=bind_address,
            grpc_port=grpc_port,
            web_port=web_port,
            auto_kill_stale_rerun_process=self.cfg.auto_kill_stale_rerun_process,
        )
        if not owns_server:
            logger.info("[RerunVisualizer] Reusing existing rerun server at %s.", rerun_address)
        if self.cfg.open_browser:
            _open_rerun_web_viewer(_normalize_host(bind_address), web_port, rerun_address)

        self._viewer = NewtonViewerRerun(
            app_id=self.cfg.app_id,
            address=rerun_address,
            serve_web_viewer=False,
            web_port=web_port,
            grpc_port=grpc_port,
            keep_historical_data=self.cfg.keep_historical_data,
            keep_scalar_history=self.cfg.keep_scalar_history,
            record_to_rrd=self.cfg.record_to_rrd,
        )
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
                ("auto_kill_stale_rerun_process", self.cfg.auto_kill_stale_rerun_process),
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
