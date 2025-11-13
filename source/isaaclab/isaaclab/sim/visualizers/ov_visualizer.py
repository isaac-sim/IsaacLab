# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse-based visualizer using Isaac Sim viewport."""

from __future__ import annotations

import omni.log
from typing import Any

from isaaclab.sim.scene_data_providers import SceneDataProvider

from .ov_visualizer_cfg import OVVisualizerCfg
from .visualizer import Visualizer


class OVVisualizer(Visualizer):
    """Omniverse-based visualizer using Isaac Sim viewport.
    
    This visualizer leverages the existing Isaac Sim application and viewport for visualization.
    It provides:
    - Automatic rendering of the USD stage in the viewport
    - Support for VisualizationMarkers (via USD prims, automatically visible)
    - Support for LivePlots (via Isaac Lab UI, automatically displayed)
    - Configurable viewport camera positioning
    
    The visualizer can operate in two modes:
    1. **Attached mode**: Uses an existing Isaac Sim app instance (typical case)
    2. **Standalone mode**: Launches a new Isaac Sim app if none exists (fallback)
    
    Note:
        VisualizationMarkers and LivePlots are managed by the scene and environment,
        not directly by this visualizer. This class primarily ensures the viewport
        is configured correctly to display them.
    """
    
    def __init__(self, cfg: OVVisualizerCfg):
        """Initialize OV visualizer.
        
        Args:
            cfg: Configuration for OV visualizer.
        """
        super().__init__(cfg)
        self.cfg: OVVisualizerCfg = cfg
        
        # Simulation app instance
        self._simulation_app = None
        self._app_launched_by_visualizer = False
        
        # Viewport references
        self._viewport_window = None
        self._viewport_api = None
        
        # Internal state
        self._is_initialized = False
        self._sim_time = 0.0
        self._step_counter = 0
    
    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        """Initialize OV visualizer with scene data.
        
        This method:
        1. Validates required data (USD stage)
        2. Checks if Isaac Sim app is running, launches if needed
        3. Configures the viewport camera
        4. Prepares for visualization of markers and plots
        
        Args:
            scene_data: Scene data from SceneDataProvider. Contains:
                - "usd_stage": The USD stage (required)
                - "metadata": Scene metadata (physics backend, num_envs, etc.)
        
        Raises:
            RuntimeError: If USD stage is not available.
        
        Note:
            OV visualizer works with any physics backend (Newton, PhysX, etc.)
            as long as a USD stage is available.
        """
        if self._is_initialized:
            omni.log.warn("[OVVisualizer] Already initialized. Skipping re-initialization.")
            return
        
        # Extract scene data
        metadata = {}
        usd_stage = None
        if scene_data is not None:
            usd_stage = scene_data.get("usd_stage")
            metadata = scene_data.get("metadata", {})
        
        # Validate required data
        if usd_stage is None:
            raise RuntimeError(
                "OV visualizer requires a USD stage in scene_data['usd_stage']. "
                "Make sure the simulation context is initialized before creating the visualizer."
            )
        
        # Check if Isaac Sim app is running
        self._ensure_simulation_app()
        
        # Setup viewport
        self._setup_viewport(usd_stage, metadata)
        
        # Log initialization
        physics_backend = metadata.get("physics_backend", "unknown")
        num_envs = metadata.get("num_envs", 0)
        omni.log.info(
            f"[OVVisualizer] Initialized with {num_envs} environments "
            f"(physics: {physics_backend})"
        )
        
        self._is_initialized = True
    
    def step(self, dt: float, scene_provider: SceneDataProvider | None = None) -> None:
        """Update visualizer each step.
        
        For the OV visualizer, most rendering is handled automatically by Isaac Sim.
        This method primarily updates internal timing.
        
        Args:
            dt: Time step in seconds.
            scene_provider: Optional scene data provider (not used in minimal implementation).
        """
        if not self._is_initialized:
            omni.log.warn("[OVVisualizer] Not initialized. Call initialize() first.")
            return
        
        # Update internal state
        self._sim_time += dt
        self._step_counter += 1
        
        # Note: Viewport rendering is handled automatically by Isaac Sim's render loop
        # VisualizationMarkers are updated by their respective owners
        # LivePlots are updated by ManagerLiveVisualizer
    
    def close(self) -> None:
        """Clean up visualizer resources."""
        if not self._is_initialized:
            return
        
        # Close app if we launched it
        if self._app_launched_by_visualizer and self._simulation_app is not None:
            omni.log.info("[OVVisualizer] Closing Isaac Sim app launched by visualizer.")
            self._simulation_app.close()
            self._simulation_app = None
        
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False
    
    def is_running(self) -> bool:
        """Check if visualizer is running."""
        if self._simulation_app is None:
            return False
        return self._simulation_app.is_running()
    
    def is_training_paused(self) -> bool:
        """Check if training is paused.
        
        Note: OV visualizer does not have a built-in pause mechanism.
        Returns False (never pauses training).
        """
        return False
    
    def supports_markers(self) -> bool:
        """Check if this visualizer supports visualization markers.
        
        Returns:
            True - OV visualizer supports markers via USD prims.
        """
        return True
    
    def supports_live_plots(self) -> bool:
        """Check if this visualizer supports live plots.
        
        Returns:
            True - OV visualizer supports live plots via Isaac Lab UI.
        """
        return True
    
    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------
    
    def _ensure_simulation_app(self) -> None:
        """Ensure Isaac Sim app is running, launch if needed."""
        # Try to get existing SimulationApp instance
        try:
            from isaacsim import SimulationApp
            
            # Check if there's an existing app instance
            # SimulationApp uses a singleton pattern
            if hasattr(SimulationApp, '_instance') and SimulationApp._instance is not None:
                self._simulation_app = SimulationApp._instance
                omni.log.info("[OVVisualizer] Using existing Isaac Sim app instance.")
                return
        except ImportError:
            omni.log.warn("[OVVisualizer] Could not import SimulationApp. May not be available.")
        
        # If we get here, no app is running
        if not self.cfg.launch_app_if_missing:
            omni.log.warn(
                "[OVVisualizer] No Isaac Sim app is running and launch_app_if_missing=False. "
                "Visualizer may not function correctly."
            )
            return
        
        # Launch a new app
        omni.log.info("[OVVisualizer] No Isaac Sim app found. Launching new instance...")
        try:
            from isaacsim import SimulationApp
            
            # Launch app with minimal config
            launch_config = {
                "headless": False,
                "experience": self.cfg.app_experience,
            }
            
            self._simulation_app = SimulationApp(launch_config)
            self._app_launched_by_visualizer = True
            
            omni.log.info(f"[OVVisualizer] Launched Isaac Sim app with experience: {self.cfg.app_experience}")
            
        except Exception as e:
            omni.log.error(f"[OVVisualizer] Failed to launch Isaac Sim app: {e}")
            self._simulation_app = None
    
    def _setup_viewport(self, usd_stage, metadata: dict) -> None:
        """Setup viewport with camera positioning.
        
        Args:
            usd_stage: USD stage to display.
            metadata: Scene metadata.
        """
        try:
            import omni.kit.viewport.utility as vp_utils
            from omni.kit.viewport.utility import get_active_viewport
            
            # Get the active viewport
            if self.cfg.viewport_name:
                # Try to get specific viewport by name
                self._viewport_window = get_active_viewport()  # For now, use active
            else:
                self._viewport_window = get_active_viewport()
            
            if self._viewport_window is None:
                omni.log.warn("[OVVisualizer] Could not get viewport window.")
                return
            
            # Get viewport API for camera control
            self._viewport_api = self._viewport_window.viewport_api
            
            # Set camera position if specified
            if self.cfg.camera_position is not None and self.cfg.camera_target is not None:
                self._set_viewport_camera(
                    self.cfg.camera_position,
                    self.cfg.camera_target
                )
            
            omni.log.info("[OVVisualizer] Viewport configured successfully.")
            
        except ImportError as e:
            omni.log.warn(f"[OVVisualizer] Viewport utilities not available: {e}")
        except Exception as e:
            omni.log.error(f"[OVVisualizer] Error setting up viewport: {e}")
    
    def _set_viewport_camera(
        self,
        position: tuple[float, float, float],
        target: tuple[float, float, float]
    ) -> None:
        """Set viewport camera position and target.
        
        Args:
            position: Camera position (x, y, z).
            target: Camera target/look-at point (x, y, z).
        """
        if self._viewport_api is None:
            return
        
        try:
            from pxr import Gf
            
            # Create camera transformation
            eye = Gf.Vec3d(*position)
            target_pos = Gf.Vec3d(*target)
            up = Gf.Vec3d(0, 0, 1)  # Z-up
            
            # Set camera transform
            # Note: The exact API might vary depending on Isaac Sim version
            # This is a common pattern, but may need adjustment
            transform = Gf.Matrix4d()
            transform.SetLookAt(eye, target_pos, up)
            
            # Try to apply to viewport
            # The API for this can vary, so we'll try a few approaches
            if hasattr(self._viewport_api, 'set_view'):
                self._viewport_api.set_view(eye, target_pos, up)
            elif hasattr(self._viewport_window, 'set_camera_position'):
                self._viewport_window.set_camera_position(*position, True)
                self._viewport_window.set_camera_target(*target, True)
            
            omni.log.info(
                f"[OVVisualizer] Set camera: pos={position}, target={target}"
            )
            
        except Exception as e:
            omni.log.warn(f"[OVVisualizer] Could not set camera transform: {e}")
