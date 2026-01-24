# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rerun-based visualizer using rerun-sdk."""

from __future__ import annotations

import logging
from typing import Any

from .rerun_visualizer_cfg import RerunVisualizerCfg
from .visualizer import Visualizer

logger = logging.getLogger(__name__)

try:
    import rerun as rr
    import rerun.blueprint as rrb
    from newton.viewer import ViewerRerun

    _RERUN_AVAILABLE = True
except ImportError:
    rr = None
    rrb = None
    ViewerRerun = None
    _RERUN_AVAILABLE = False


class NewtonViewerRerun(ViewerRerun if _RERUN_AVAILABLE else object):
    """Isaac Lab wrapper for Newton's ViewerRerun."""

    def __init__(
        self,
        app_id: str | None = None,
        web_port: int | None = None,
        keep_historical_data: bool = False,
        keep_scalar_history: bool = True,
        record_to_rrd: str | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(
            app_id=app_id,
            web_port=web_port,
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
    """Rerun web-based visualizer with time scrubbing, recording, and data inspection."""

    def __init__(self, cfg: RerunVisualizerCfg):
        super().__init__(cfg)
        self.cfg: RerunVisualizerCfg = cfg

        if not _RERUN_AVAILABLE:
            raise ImportError("Rerun visualizer requires rerun-sdk and Newton. Install: pip install rerun-sdk")

        self._viewer: NewtonViewerRerun | None = None
        self._model = None
        self._state = None
        self._is_initialized = False
        self._sim_time = 0.0
        self._scene_data_provider = None

    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        if self._is_initialized:
            logger.warning("[RerunVisualizer] Already initialized. Skipping re-initialization.")
            return

        if scene_data and "scene_data_provider" in scene_data:
            self._scene_data_provider = scene_data["scene_data_provider"]

        metadata = {}
        if self._scene_data_provider:
            self._model = self._scene_data_provider.get_model()
            self._state = self._scene_data_provider.get_state()
            metadata = self._scene_data_provider.get_metadata()
        else:
            try:
                from isaaclab.sim._impl.newton_manager import NewtonManager

                self._model = NewtonManager._model
                self._state = NewtonManager._state_0
                metadata = {
                    "physics_backend": "newton",
                    "num_envs": NewtonManager._num_envs if NewtonManager._num_envs is not None else 0,
                    "gravity_vector": NewtonManager._gravity_vector,
                    "clone_physics_only": NewtonManager._clone_physics_only,
                }
            except Exception:
                pass

        if self._model is None:
            raise RuntimeError(
                "Rerun visualizer requires a Newton Model. Ensure a scene data provider is available."
            )

        if self._state is None:
            logger.warning("[RerunVisualizer] No Newton State available. Visualization may not work correctly.")

        try:
            if self.cfg.record_to_rrd:
                logger.info(f"[RerunVisualizer] Recording enabled to: {self.cfg.record_to_rrd}")

            self._viewer = NewtonViewerRerun(
                app_id=self.cfg.app_id,
                web_port=self.cfg.web_port,
                keep_historical_data=self.cfg.keep_historical_data,
                keep_scalar_history=self.cfg.keep_scalar_history,
                record_to_rrd=self.cfg.record_to_rrd,
                metadata=metadata,
            )

            self._viewer.set_model(self._model)
            self._viewer.set_world_offsets((0.0, 0.0, 0.0))

            try:
                cam_pos = self.cfg.camera_position
                cam_target = self.cfg.camera_target

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

                logger.info(f"[RerunVisualizer] Set initial camera view: position={cam_pos}, target={cam_target}")
            except Exception as exc:
                logger.warning(f"[RerunVisualizer] Could not set initial camera view: {exc}")

            num_envs = metadata.get("num_envs", 0)
            physics_backend = metadata.get("physics_backend", "unknown")
            logger.info(f"[RerunVisualizer] Initialized with {num_envs} environments (physics: {physics_backend})")

            self._is_initialized = True
        except Exception as exc:
            logger.error(f"[RerunVisualizer] Failed to initialize viewer: {exc}")
            raise

    def step(self, dt: float, state: Any | None = None) -> None:
        if not self._is_initialized or self._viewer is None:
            logger.warning("[RerunVisualizer] Not initialized. Call initialize() first.")
            return

        if self._scene_data_provider:
            self._state = self._scene_data_provider.get_state()
        else:
            try:
                from isaaclab.sim._impl.newton_manager import NewtonManager

                self._state = NewtonManager._state_0
            except Exception:
                self._state = None

        self._sim_time += dt

        self._viewer.begin_frame(self._sim_time)
        if self._state is not None:
            self._viewer.log_state(self._state)
        self._viewer.end_frame()

    def close(self) -> None:
        if not self._is_initialized or self._viewer is None:
            return

        try:
            if self.cfg.record_to_rrd:
                logger.info(f"[RerunVisualizer] Finalizing recording to: {self.cfg.record_to_rrd}")
            self._viewer.close()
            logger.info("[RerunVisualizer] Closed successfully.")
            if self.cfg.record_to_rrd:
                import os

                if os.path.exists(self.cfg.record_to_rrd):
                    size = os.path.getsize(self.cfg.record_to_rrd)
                    logger.info(f"[RerunVisualizer] Recording saved: {self.cfg.record_to_rrd} ({size} bytes)")
                else:
                    logger.warning(f"[RerunVisualizer] Recording file not found: {self.cfg.record_to_rrd}")
        except Exception as exc:
            logger.warning(f"[RerunVisualizer] Error during close: {exc}")

        self._viewer = None
        self._is_initialized = False

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
