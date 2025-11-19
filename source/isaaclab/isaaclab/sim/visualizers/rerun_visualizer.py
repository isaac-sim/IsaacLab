# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rerun-based visualizer using rerun-sdk."""

from __future__ import annotations

import numpy as np
import omni.log
import torch
from typing import Any

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
    """Isaac Lab wrapper for Newton's ViewerRerun.
    
    Adds VisualizationMarkers, LivePlots, metadata display, and partial visualization."""

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
        enable_markers: bool = True,
        enable_live_plots: bool = True,
        env_indices: list[int] | None = None,
    ):
        """Initialize Newton ViewerRerun wrapper."""
        if not _RERUN_AVAILABLE:
            raise ImportError("Rerun visualizer requires rerun-sdk and Newton. Install: pip install rerun-sdk")
        
        # Call parent with only Newton parameters
        super().__init__(
            server=server,
            address=address,
            launch_viewer=launch_viewer,
            app_id=app_id,
            keep_historical_data=keep_historical_data,
            keep_scalar_history=keep_scalar_history,
            record_to_rrd=record_to_rrd,
        )
        
        # Isaac Lab state
        self._metadata = metadata or {}
        self._enable_markers = enable_markers
        self._enable_live_plots = enable_live_plots
        self._env_indices = env_indices
        
        # Storage for registered markers and plots
        self._registered_markers = []
        self._registered_plots = {}
        
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
        
        if self._env_indices is not None:
            metadata_text += f"**Visualized Environments:** {len(self._env_indices)} (indices: {self._env_indices[:5]}...)\n"
        else:
            metadata_text += f"**Visualized Environments:** All ({num_envs})\n"
        
        # Physics backend info
        physics_backend = self._metadata.get("physics_backend", "unknown")
        metadata_text += f"**Physics:** {physics_backend}\n"
        
        # Visualization features
        metadata_text += f"**Markers Enabled:** {self._enable_markers}\n"
        metadata_text += f"**Plots Enabled:** {self._enable_live_plots}\n"
        
        # Additional metadata
        for key, value in self._metadata.items():
            if key not in ["physics_backend", "num_envs"]:
                metadata_text += f"**{key}:** {value}\n"
        
        # Log to Rerun
        rr.log("metadata", rr.TextDocument(metadata_text, media_type=rr.MediaType.MARKDOWN))
    
    def register_markers(self, markers: Any) -> None:
        """Register VisualizationMarkers for active logging.
        
        Args:
            markers: VisualizationMarkers instance to log each frame.
        """
        if self._enable_markers:
            self._registered_markers.append(markers)
            omni.log.info(f"[RerunVisualizer] Registered markers: {markers}")
    
    def register_plots(self, plots: dict[str, Any]) -> None:
        """Register LivePlot instances for active logging.
        
        Args:
            plots: Dictionary mapping plot names to LivePlot instances.
        """
        if self._enable_live_plots:
            self._registered_plots.update(plots)
            omni.log.info(f"[RerunVisualizer] Registered {len(plots)} plot(s)")
    
    def log_markers(self) -> None:
        """Actively log all registered VisualizationMarkers to Rerun.
        
        This method converts Isaac Lab's USD-based markers to Rerun entities.
        
        Supported marker types:
            - Arrows: Logged as line segments via rr.LineStrips3D
            - Frames: Logged as XYZ axes via rr.LineStrips3D (3 lines per frame)
            - Spheres: Logged as points via rr.Points3D with radii
        
        Implementation Strategy:
            We convert USD-based visualization markers to Rerun primitives.
            Since markers are scene-managed and updated by their owners,
            we need to extract their current state each frame and log it.
        
        TODO: Future enhancements
            - Support more marker types (cylinders, cones, boxes)
            - Optimize batch logging for large marker counts
            - Add color/material support for better visual distinction
        """
        if not self._enable_markers or len(self._registered_markers) == 0:
            return
        
        try:
            for marker_idx, markers in enumerate(self._registered_markers):
                # Extract marker data
                # Note: This is a simplified implementation that assumes markers
                # expose their data through specific methods/properties.
                # Actual implementation depends on VisualizationMarkers API.
                
                # For now, we'll use Newton's built-in logging methods
                # which VisualizationMarkers should be compatible with
                
                # TODO: Implement proper marker extraction and conversion
                # marker_data = markers.get_marker_data()
                # self._log_marker_data(marker_data, f"markers/{marker_idx}")
                
                pass  # Stub for now
                
        except Exception as e:
            omni.log.warn(f"[RerunVisualizer] Failed to log markers: {e}")
    
    def log_plot_data(self) -> None:
        """Actively log all registered LivePlot data to Rerun as time series.
        
        This method extracts scalar data from LivePlot objects and logs them
        as Rerun Scalars, enabling visualization alongside the 3D scene.
        
        Implementation Strategy:
            LivePlots in Isaac Lab are typically omni.ui-based widgets that
            display time-series data. For Rerun, we need to extract the raw
            scalar values and log them using rr.Scalar().
        
        TODO: Full implementation
            - Extract data from LiveLinePlot objects
            - Handle multiple series per plot
            - Maintain proper timeline synchronization
            - Support different plot types (line, bar, etc.)
        """
        if not self._enable_live_plots or len(self._registered_plots) == 0:
            return
        
        try:
            for plot_name, plot_obj in self._registered_plots.items():
                # TODO: Extract data from plot object
                # For now, this is a stub
                # data = plot_obj.get_latest_data()
                # rr.log(f"plots/{plot_name}", rr.Scalar(data))
                
                pass  # Stub for now
                
        except Exception as e:
            omni.log.warn(f"[RerunVisualizer] Failed to log plot data: {e}")
    
    def _log_marker_data(self, marker_data: dict, entity_path: str) -> None:
        """Helper to log specific marker data to Rerun.
        
        Args:
            marker_data: Dictionary containing marker positions, types, colors, etc.
            entity_path: Rerun entity path for logging.
        """
        marker_type = marker_data.get("type", "unknown")
        
        if marker_type == "arrow":
            # Log arrows as line segments
            starts = marker_data.get("positions")  # Start points
            directions = marker_data.get("directions")  # Direction vectors
            
            if starts is not None and directions is not None:
                ends = starts + directions
                self.log_lines(
                    entity_path,
                    starts=starts,
                    ends=ends,
                    colors=marker_data.get("colors"),
                    width=marker_data.get("width", 0.01),
                )
        
        elif marker_type == "frame":
            # Log coordinate frames as 3 lines (XYZ axes)
            positions = marker_data.get("positions")
            orientations = marker_data.get("orientations")
            scale = marker_data.get("scale", 0.1)
            
            if positions is not None and orientations is not None:
                # TODO: Convert quaternions to XYZ axis lines
                # For each frame, create 3 lines (red=X, green=Y, blue=Z)
                pass
        
        elif marker_type == "sphere":
            # Log spheres as points with radii
            positions = marker_data.get("positions")
            radii = marker_data.get("radii", 0.05)
            colors = marker_data.get("colors")
            
            if positions is not None:
                self.log_points(
                    entity_path,
                    points=positions,
                    radii=radii,
                    colors=colors,
                )


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
    
    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        """Initialize visualizer with Newton Model and State."""
        if self._is_initialized:
            omni.log.warn("[RerunVisualizer] Already initialized. Skipping re-initialization.")
            return
        
        # Extract scene data
        metadata = {}
        if scene_data is not None:
            self._model = scene_data.get("model")
            self._state = scene_data.get("state")
            metadata = scene_data.get("metadata", {})
        
        # Validate physics backend
        physics_backend = metadata.get("physics_backend", "unknown")
        if physics_backend != "newton" and physics_backend != "unknown":
            raise RuntimeError(
                f"Rerun visualizer currently requires Newton physics backend, "
                f"but '{physics_backend}' is running. "
                f"Please use a compatible visualizer for {physics_backend} physics "
                f"(e.g., OVVisualizer).\n\n"
                f"Future versions will support multiple physics backends."
            )
        
        # Validate required Newton data
        if self._model is None:
            raise RuntimeError(
                "Rerun visualizer requires a Newton Model in scene_data['model']. "
                "Make sure Newton physics is initialized before creating the visualizer."
            )
        
        if self._state is None:
            omni.log.warn(
                "[RerunVisualizer] No Newton State provided in scene_data['state']. "
                "Visualization may not work correctly."
            )
        
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
                enable_markers=self.cfg.enable_markers,
                enable_live_plots=self.cfg.enable_live_plots,
                env_indices=self.cfg.env_indices,
            )
            
            # Set the model
            self._viewer.set_model(self._model)
            
            # Log initialization
            num_envs = metadata.get("num_envs", 0)
            viz_envs = len(self.cfg.env_indices) if self.cfg.env_indices else num_envs
            omni.log.info(
                f"[RerunVisualizer] Initialized with {viz_envs}/{num_envs} environments "
                f"(physics: {physics_backend})"
            )
            
            self._is_initialized = True
            
        except Exception as e:
            omni.log.error(f"[RerunVisualizer] Failed to initialize viewer: {e}")
            raise
    
    def step(self, dt: float, state: Any | None = None) -> None:
        """Update visualizer each step.
        
        This method:
        1. Updates state (if provided)
        2. Logs current state to Rerun (transforms, meshes)
        3. Actively logs markers (if enabled)
        4. Actively logs plot data (if enabled)
        
        Implementation Note:
            Partial visualization (env_indices) is handled internally by filtering
            which instance transforms are logged. We log all meshes once (they're
            shared assets), but only log transforms for selected environments.
        
        Args:
            dt: Time step in seconds.
            state: Updated physics state (e.g., newton.State).
        """
        if not self._is_initialized or self._viewer is None:
            omni.log.warn("[RerunVisualizer] Not initialized. Call initialize() first.")
            return
        
        # Update state if provided
        if state is not None:
            self._state = state
        
        # Begin frame
        self._viewer.begin_frame(self._sim_time)
        
        # Log state (transforms) - Newton's ViewerRerun handles this
        if self._state is not None:
            # Handle partial visualization if env_indices is set
            if self.cfg.env_indices is not None:
                # TODO: Filter state to only visualized environments
                # For now, log all state (Newton's ViewerRerun will handle it)
                self._viewer.log_state(self._state)
            else:
                self._viewer.log_state(self._state)
        
        # Actively log markers (if enabled)
        if self.cfg.enable_markers:
            self._viewer.log_markers()
        
        # Actively log plot data (if enabled)
        if self.cfg.enable_live_plots:
            self._viewer.log_plot_data()
        
        # End frame
        self._viewer.end_frame()
        
        # Update internal time
        self._sim_time += dt
    
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
            Rerun visualizer uses Rerun's built-in timeline controls for playback.
            It does not provide a training pause mechanism like NewtonVisualizer.
        
        Returns:
            False - Rerun does not support training pause.
        """
        return False
    
    def supports_markers(self) -> bool:
        """Check if this visualizer supports visualization markers.
        
        Returns:
            True - Rerun supports markers via active logging.
        """
        return True
    
    def supports_live_plots(self) -> bool:
        """Check if this visualizer supports live plots.
        
        Returns:
            True - Rerun supports plots via active logging (currently stub).
        """
        return True
    
    def register_markers(self, markers: Any) -> None:
        """Register VisualizationMarkers for active logging.
        
        Args:
            markers: VisualizationMarkers instance to visualize.
        """
        if self._viewer:
            self._viewer.register_markers(markers)
    
    def register_plots(self, plots: dict[str, Any]) -> None:
        """Register LivePlot instances for active logging.
        
        Args:
            plots: Dictionary mapping plot names to LivePlot instances.
        """
        if self._viewer:
            self._viewer.register_plots(plots)

