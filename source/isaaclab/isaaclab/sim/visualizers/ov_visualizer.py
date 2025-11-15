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
    """Omniverse visualizer using Isaac Sim viewport.
    
    Renders USD stage with VisualizationMarkers and LivePlots.
    Can attach to existing app or launch standalone.
    """
    
    def __init__(self, cfg: OVVisualizerCfg):
        super().__init__(cfg)
        self.cfg: OVVisualizerCfg = cfg
        
        self._simulation_app = None
        self._app_launched_by_visualizer = False
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False
        self._sim_time = 0.0
        self._step_counter = 0
    
    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        """Initialize OV visualizer."""
        if self._is_initialized:
            omni.log.warn("[OVVisualizer] Already initialized.")
            return
        
        metadata = {}
        usd_stage = None
        if scene_data is not None:
            usd_stage = scene_data.get("usd_stage")
            metadata = scene_data.get("metadata", {})
        
        if usd_stage is None:
            raise RuntimeError("OV visualizer requires a USD stage.")
        
        self._ensure_simulation_app()
        self._setup_viewport(usd_stage, metadata)
        
        physics_backend = metadata.get("physics_backend", "unknown")
        num_envs = metadata.get("num_envs", 0)
        omni.log.info(f"[OVVisualizer] Initialized ({num_envs} envs, {physics_backend} physics)")
        
        self._is_initialized = True
    
    def step(self, dt: float, scene_provider: SceneDataProvider | None = None) -> None:
        """Update visualizer (rendering handled automatically by Isaac Sim)."""
        if not self._is_initialized:
            return
        self._sim_time += dt
        self._step_counter += 1
    
    def close(self) -> None:
        """Clean up visualizer resources."""
        if not self._is_initialized:
            return
        
        if self._app_launched_by_visualizer and self._simulation_app is not None:
            omni.log.info("[OVVisualizer] Closing Isaac Sim app.")
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
        """Check if training is paused (always False for OV)."""
        return False
    
    def supports_markers(self) -> bool:
        """Supports markers via USD prims."""
        return True
    
    def supports_live_plots(self) -> bool:
        """Supports live plots via Isaac Lab UI."""
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
        """Setup viewport with camera and window size."""
        try:
            import omni.kit.viewport.utility as vp_utils
            
            # Create new viewport or use existing
            if self.cfg.create_viewport and self.cfg.viewport_name:
                # Create new viewport with proper API
                self._viewport_window = vp_utils.create_viewport_window(
                    name=self.cfg.viewport_name,
                    width=self.cfg.window_width,
                    height=self.cfg.window_height,
                    position_x=50,  # Window position on screen
                    position_y=50,
                )
                omni.log.info(f"[OVVisualizer] Created viewport '{self.cfg.viewport_name}'")
            else:
                # Use existing active viewport
                self._viewport_window = vp_utils.get_active_viewport_window()
                omni.log.info("[OVVisualizer] Using existing active viewport")
            
            if self._viewport_window is None:
                omni.log.warn("[OVVisualizer] Could not get/create viewport.")
                return
            
            # Get viewport API for camera control
            self._viewport_api = self._viewport_window.viewport_api
            
            # Set camera pose using Isaac Sim utility
            self._set_viewport_camera(self.cfg.camera_position, self.cfg.camera_target)
            
            omni.log.info(f"[OVVisualizer] Viewport configured (size: {self.cfg.window_width}x{self.cfg.window_height})")
            
        except ImportError as e:
            omni.log.warn(f"[OVVisualizer] Viewport utilities unavailable: {e}")
        except Exception as e:
            omni.log.error(f"[OVVisualizer] Error setting up viewport: {e}")
    
    def _set_viewport_camera(
        self,
        position: tuple[float, float, float],
        target: tuple[float, float, float]
    ) -> None:
        """Set viewport camera position and target using Isaac Sim utilities."""
        if self._viewport_api is None:
            return
        
        try:
            # Import Isaac Sim viewport utilities
            import isaacsim.core.utils.viewports as vp_utils
            
            # Get the camera prim path for this viewport
            camera_path = self._viewport_api.get_active_camera()
            if not camera_path:
                camera_path = "/OmniverseKit_Persp"  # Default camera
            
            # Use Isaac Sim utility to set camera view
            vp_utils.set_camera_view(
                eye=list(position),
                target=list(target),
                camera_prim_path=camera_path,
                viewport_api=self._viewport_api
            )
            
            omni.log.info(f"[OVVisualizer] Camera set: pos={position}, target={target}, camera={camera_path}")
            
        except Exception as e:
            omni.log.warn(f"[OVVisualizer] Could not set camera: {e}")
