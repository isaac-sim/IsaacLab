# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rerun-based visualizer using rerun-sdk."""

from __future__ import annotations

from typing import Any

import omni.log

from .rerun_visualizer_cfg import RerunVisualizerCfg
from .visualizer import Visualizer

# Try to import rerun and Newton's ViewerRerun
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
        server: bool = True,
        address: str = "127.0.0.1:9876",
        launch_viewer: bool = True,
        app_id: str | None = None,
        keep_historical_data: bool = False,
        keep_scalar_history: bool = True,
        record_to_rrd: str | None = None,
        metadata: dict | None = None,
    ):
        """Initialize Newton ViewerRerun wrapper."""
        # Call parent with only Newton parameters
        super().__init__(
            server=server,
            address=address,
            launch_viewer=launch_viewer,
            app_id=app_id,
            # Note: The current Newton version with IsaacLab does not support these recording flags.
            # Support is available in the top of tree Newton version when we eventually upgrade.
            # keep_historical_data=keep_historical_data,
            # keep_scalar_history=keep_scalar_history,
            # record_to_rrd=record_to_rrd,
        )

        # Isaac Lab state
        self._metadata = metadata or {}

        # Log metadata on initialization
        self._log_metadata()

    def _log_metadata(self) -> None:
        """Log scene metadata to Rerun as text."""
        metadata_text = "# Isaac Lab Scene Metadata\n\n"

        # Physics info
        physics_backend = self._metadata.get("physics_backend", "unknown")
        metadata_text += f"**Physics Backend:** {physics_backend}\n"

        # Environment info
        num_envs = self._metadata.get("num_envs", 0)
        metadata_text += f"**Total Environments:** {num_envs}\n"

        # Additional metadata
        for key, value in self._metadata.items():
            if key not in ["physics_backend", "num_envs"]:
                metadata_text += f"**{key}:** {value}\n"

        # Log to Rerun
        rr.log("metadata", rr.TextDocument(metadata_text, media_type=rr.MediaType.MARKDOWN))


class RerunVisualizer(Visualizer):
    """Rerun web-based visualizer with time scrubbing, recording, and data inspection.

    Requires Newton physics backend and rerun-sdk (pip install rerun-sdk)."""

    def __init__(self, cfg: RerunVisualizerCfg):
        """Initialize Rerun visualizer."""
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
        """Initialize visualizer with Newton Model and State."""
        if self._is_initialized:
            omni.log.warn("[RerunVisualizer] Already initialized. Skipping re-initialization.")
            return

        # Import NewtonManager for metadata access
        from isaaclab.sim._impl.newton_manager import NewtonManager

        # Store scene data provider for accessing physics state
        if scene_data and "scene_data_provider" in scene_data:
            self._scene_data_provider = scene_data["scene_data_provider"]

        # Get Newton-specific data from scene data provider
        if self._scene_data_provider:
            self._model = self._scene_data_provider.get_model()
            self._state = self._scene_data_provider.get_state()
        else:
            # Fallback: direct access to NewtonManager (for backward compatibility)
            self._model = NewtonManager._model
            self._state = NewtonManager._state_0

        # Validate required Newton data
        if self._model is None:
            raise RuntimeError(
                "Rerun visualizer requires a Newton Model. "
                "Make sure Newton physics is initialized before creating the visualizer."
            )

        if self._state is None:
            omni.log.warn("[RerunVisualizer] No Newton State available. Visualization may not work correctly.")

        # Build metadata from NewtonManager
        metadata = {
            "physics_backend": "newton",
            "num_envs": NewtonManager._num_envs if NewtonManager._num_envs is not None else 0,
            "gravity_vector": NewtonManager._gravity_vector,
            "clone_physics_only": NewtonManager._clone_physics_only,
        }

        # Create Newton ViewerRerun wrapper
        try:
            if self.cfg.record_to_rrd:
                omni.log.info(f"[RerunVisualizer] Recording enabled to: {self.cfg.record_to_rrd}")

            self._viewer = NewtonViewerRerun(
                server=self.cfg.server_mode,
                address=self.cfg.server_address,
                launch_viewer=self.cfg.launch_viewer,
                app_id=self.cfg.app_id,
                keep_historical_data=self.cfg.keep_historical_data,
                keep_scalar_history=self.cfg.keep_scalar_history,
                record_to_rrd=self.cfg.record_to_rrd,
                metadata=metadata,
            )

            # Set the model
            self._viewer.set_model(self._model)

            # Log initialization
            num_envs = metadata.get("num_envs", 0)
            physics_backend = metadata.get("physics_backend", "newton")
            omni.log.info(f"[RerunVisualizer] Initialized with {num_envs} environments (physics: {physics_backend})")

            self._is_initialized = True

        except Exception as e:
            omni.log.error(f"[RerunVisualizer] Failed to initialize viewer: {e}")
            raise

    def step(self, dt: float, state: Any | None = None) -> None:
        """Update visualizer each step.

        This method:
        1. Fetches updated state from NewtonManager
        2. Logs current state to Rerun (transforms, meshes)
        3. Actively logs markers (if enabled)
        4. Actively logs plot data (if enabled)

        Args:
            dt: Time step in seconds.
            state: Unused (deprecated parameter, kept for API compatibility).
        """
        if not self._is_initialized or self._viewer is None:
            omni.log.warn("[RerunVisualizer] Not initialized. Call initialize() first.")
            return

        # Fetch updated state from scene data provider
        if self._scene_data_provider:
            self._state = self._scene_data_provider.get_state()
        else:
            # Fallback: direct access to NewtonManager
            from isaaclab.sim._impl.newton_manager import NewtonManager

            self._state = NewtonManager._state_0

        # Update internal time
        self._sim_time += dt

        # Begin frame with current simulation time
        self._viewer.begin_frame(self._sim_time)

        # Log state (transforms) - Newton's ViewerRerun handles this
        if self._state is not None:
            self._viewer.log_state(self._state)

        # End frame
        self._viewer.end_frame()

    def close(self) -> None:
        """Clean up Rerun visualizer resources and finalize recordings."""
        if not self._is_initialized or self._viewer is None:
            return

        try:
            if self.cfg.record_to_rrd:
                omni.log.info(f"[RerunVisualizer] Finalizing recording to: {self.cfg.record_to_rrd}")
            self._viewer.close()
            omni.log.info("[RerunVisualizer] Closed successfully.")
            if self.cfg.record_to_rrd:
                import os

                if os.path.exists(self.cfg.record_to_rrd):
                    size = os.path.getsize(self.cfg.record_to_rrd)
                    omni.log.info(f"[RerunVisualizer] Recording saved: {self.cfg.record_to_rrd} ({size} bytes)")
                else:
                    omni.log.warn(f"[RerunVisualizer] Recording file not found: {self.cfg.record_to_rrd}")
        except Exception as e:
            omni.log.warn(f"[RerunVisualizer] Error during close: {e}")

        self._viewer = None
        self._is_initialized = False

    def is_running(self) -> bool:
        """Check if visualizer is running.

        Returns:
            True if viewer is initialized and running, False otherwise.
        """
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def is_training_paused(self) -> bool:
        """Check if training is paused.

        Note:
            Rerun visualizer currently uses Rerun's built-in timeline controls for playback.
        """
        return False

    def supports_markers(self) -> bool:
        """Rerun visualizer does not have this feature yet."""
        return False

    def supports_live_plots(self) -> bool:
        """Rerun visualizer does not have this feature yet."""
        return False
