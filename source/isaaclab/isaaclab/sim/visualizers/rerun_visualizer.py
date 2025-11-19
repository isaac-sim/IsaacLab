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
        env_ids_to_viz: list[int] | None = None,
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
            # keep_historical_data=keep_historical_data,
            # keep_scalar_history=keep_scalar_history,
            # record_to_rrd=record_to_rrd,
        )
        
        # Isaac Lab state
        self._metadata = metadata or {}
        self._enable_markers = enable_markers
        self._enable_live_plots = enable_live_plots
        self._env_ids_to_viz = env_ids_to_viz
        
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
        
        if self._env_ids_to_viz is not None:
            metadata_text += f"**Visualized Environments:** {len(self._env_ids_to_viz)} (IDs: {self._env_ids_to_viz[:5]}...)\n"
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
        """Log registered VisualizationMarkers to Rerun.
        
        Converts Isaac Lab markers to Rerun primitives (arrows, frames, spheres).
        """
        if not self._enable_markers or len(self._registered_markers) == 0:
            return
        
        try:
            for marker_idx, markers in enumerate(self._registered_markers):
                # Check if markers object has data access methods
                if not hasattr(markers, 'data'):
                    continue
                
                marker_data = markers.data
                entity_path = f"markers/{markers.__class__.__name__}_{marker_idx}"
                
                # Log arrows as line segments
                if hasattr(marker_data, 'arrows') and marker_data.arrows is not None:
                    arrows = marker_data.arrows
                    if hasattr(arrows, 'positions') and hasattr(arrows, 'directions'):
                        positions = arrows.positions.cpu().numpy() if hasattr(arrows.positions, 'cpu') else arrows.positions
                        directions = arrows.directions.cpu().numpy() if hasattr(arrows.directions, 'cpu') else arrows.directions
                        
                        if len(positions) > 0:
                            # Create line segments from position to position+direction
                            for i in range(len(positions)):
                                start = positions[i]
                                end = start + directions[i]
                                rr.log(f"{entity_path}/arrow_{i}", rr.Arrows3D(
                                    origins=[start],
                                    vectors=[directions[i]]
                                ))
                
                # Log spheres as 3D points with radii
                if hasattr(marker_data, 'spheres') and marker_data.spheres is not None:
                    spheres = marker_data.spheres
                    if hasattr(spheres, 'positions'):
                        positions = spheres.positions.cpu().numpy() if hasattr(spheres.positions, 'cpu') else spheres.positions
                        radii = spheres.radii.cpu().numpy() if hasattr(spheres, 'radii') else [0.05] * len(positions)
                        
                        if len(positions) > 0:
                            rr.log(f"{entity_path}/spheres", rr.Points3D(
                                positions=positions,
                                radii=radii
                            ))
                
                # Log coordinate frames as transform axes
                if hasattr(marker_data, 'frames') and marker_data.frames is not None:
                    frames = marker_data.frames
                    if hasattr(frames, 'positions') and hasattr(frames, 'orientations'):
                        positions = frames.positions.cpu().numpy() if hasattr(frames.positions, 'cpu') else frames.positions
                        orientations = frames.orientations.cpu().numpy() if hasattr(frames.orientations, 'cpu') else frames.orientations
                        scale = frames.scale if hasattr(frames, 'scale') else 0.1
                        
                        for i in range(len(positions)):
                            # Log as transform with axes
                            rr.log(f"{entity_path}/frame_{i}", rr.Transform3D(
                                translation=positions[i],
                                rotation=rr.Quaternion(xyzw=orientations[i]),
                                axis_length=scale
                            ))
                
        except Exception as e:
            omni.log.warn(f"[RerunVisualizer] Failed to log markers: {e}")
    
    def log_plot_data(self) -> None:
        """Log registered LivePlot data to Rerun as time series scalars."""
        if not self._enable_live_plots or len(self._registered_plots) == 0:
            return
        
        try:
            for plot_name, plot_obj in self._registered_plots.items():
                # Try to extract data from common plot object attributes
                data_value = None
                
                # Method 1: Check for 'value' or 'data' attribute
                if hasattr(plot_obj, 'value'):
                    data_value = plot_obj.value
                elif hasattr(plot_obj, 'data'):
                    data_value = plot_obj.data
                
                # Method 2: Check for get_data() or get_value() methods
                elif hasattr(plot_obj, 'get_data'):
                    data_value = plot_obj.get_data()
                elif hasattr(plot_obj, 'get_value'):
                    data_value = plot_obj.get_value()
                
                # Method 3: Check for buffer/history attribute (get latest value)
                elif hasattr(plot_obj, 'buffer') and plot_obj.buffer is not None:
                    if len(plot_obj.buffer) > 0:
                        data_value = plot_obj.buffer[-1]
                elif hasattr(plot_obj, 'history') and plot_obj.history is not None:
                    if len(plot_obj.history) > 0:
                        data_value = plot_obj.history[-1]
                
                # Log the scalar value if we found it
                if data_value is not None:
                    # Convert tensor to scalar if needed
                    if hasattr(data_value, 'item'):
                        data_value = data_value.item()
                    elif hasattr(data_value, 'cpu'):
                        data_value = data_value.cpu().numpy()
                    
                    # Handle numpy arrays (take mean if multi-dimensional)
                    if isinstance(data_value, np.ndarray):
                        if data_value.size == 1:
                            data_value = float(data_value.flat[0])
                        else:
                            data_value = float(np.mean(data_value))
                    
                    # Log as scalar
                    rr.log(f"plots/{plot_name}", rr.Scalar(float(data_value)))
                
        except Exception as e:
            omni.log.warn(f"[RerunVisualizer] Failed to log plot data: {e}")
    


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
        
        # Fetch Newton-specific data from NewtonManager
        from isaaclab.sim._impl.newton_manager import NewtonManager
        
        self._model = NewtonManager._model
        self._state = NewtonManager._state_0
        
        # Validate required Newton data
        if self._model is None:
            raise RuntimeError(
                "Rerun visualizer requires a Newton Model. "
                "Make sure Newton physics is initialized before creating the visualizer."
            )
        
        if self._state is None:
            omni.log.warn(
                "[RerunVisualizer] No Newton State available. "
                "Visualization may not work correctly."
            )
        
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
                enable_markers=self.cfg.enable_markers,
                enable_live_plots=self.cfg.enable_live_plots,
                env_ids_to_viz=self.cfg.env_ids_to_viz,
            )
            
            # Set the model
            self._viewer.set_model(self._model)
            
            # Setup partial visualization (env_ids_to_viz filtering)
            num_envs = metadata.get("num_envs", 0)
            if self.cfg.env_ids_to_viz is not None:
                self._setup_env_filtering(num_envs)
            
            # Log initialization
            viz_envs = len(self.cfg.env_ids_to_viz) if self.cfg.env_ids_to_viz else num_envs
            physics_backend = metadata.get("physics_backend", "newton")
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
        1. Fetches updated state from NewtonManager
        2. Logs current state to Rerun (transforms, meshes)
        3. Actively logs markers (if enabled)
        4. Actively logs plot data (if enabled)
        
        Implementation Note:
            Partial visualization (env_ids_to_viz) is handled internally by filtering
            which instance transforms are logged. We log all meshes once (they're
            shared assets), but only log transforms for selected environments.
        
        Args:
            dt: Time step in seconds.
            state: Unused (deprecated parameter, kept for API compatibility).
        """e
        
        if not self._is_initialized or self._viewer is None:
            omni.log.warn("[RerunVisualizer] Not initialized. Call initialize() first.")
            return
        
        # Fetch updated state from NewtonManager
        from isaaclab.sim._impl.newton_manager import NewtonManager
        self._state = NewtonManager._state_0
        
        # Update internal time
        self._sim_time += dt
        
        # Begin frame with current simulation time
        self._viewer.begin_frame(self._sim_time)
        
        # Log state (transforms) - Newton's ViewerRerun handles this
        if self._state is not None:
            self._viewer.log_state(self._state)
        
        # Actively log markers (if enabled)
        if self.cfg.enable_markers:
            self._viewer.log_markers()
        
        # Actively log plot data (if enabled)
        if self.cfg.enable_live_plots:
            self._viewer.log_plot_data()
        
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
    
    def _setup_env_filtering(self, num_envs: int) -> None:
        """Setup environment filtering using world offsets.
        
        NOTE: This uses visualization-only offsets that do NOT affect physics simulation.
        Newton's world_offsets only shift the rendered/logged position of environments, not their
        physical positions. This is confirmed by Newton's test_visual_separation test.
        
        Current approach: Moves non-visualized environments far away (10000 units) to hide them.
        This works but is not ideal. Future improvements could include:
        - Proper filtering at the logging level (only log selected env transforms)
        - Custom logging callbacks for partial visualization
        - State slicing before logging
        
        Args:
            num_envs: Total number of environments.
        """
        import warp as wp
        
        # Create world offsets array
        offsets = wp.zeros(num_envs, dtype=wp.vec3, device=self._viewer.device)
        offsets_np = offsets.numpy()
        
        # Move non-visualized environments far away (visualization-only, doesn't affect physics)
        visualized_set = set(self.cfg.env_ids_to_viz)
        for world_idx in range(num_envs):
            if world_idx not in visualized_set:
                offsets_np[world_idx] = (10000.0, 10000.0, 10000.0)
        
        offsets.assign(offsets_np)
        self._viewer.world_offsets = offsets
        
        omni.log.info(
            f"[RerunVisualizer] Partial visualization enabled: "
            f"{len(self.cfg.env_ids_to_viz)}/{num_envs} environments"
        )

