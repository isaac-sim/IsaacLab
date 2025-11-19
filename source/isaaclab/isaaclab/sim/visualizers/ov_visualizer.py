# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse-based visualizer using Isaac Sim viewport."""

from __future__ import annotations

import asyncio
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
        
        # Note: We don't close the SimulationApp here as it's managed by AppLauncher
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
        """Ensure Isaac Sim app is running.
        
        The OV visualizer requires Isaac Sim to be launched via AppLauncher before initialization.
        In typical usage (e.g., training scripts), AppLauncher handles this automatically.
        
        Note: Future enhancement could add standalone app launching for non-training workflows,
        but this is not currently needed as AppLauncher is always used in practice.
        """
        try:
            # Check if omni.kit.app is available (indicates Isaac Sim is running)
            import omni.kit.app
            
            # Get the running app instance
            app = omni.kit.app.get_app()
            if app is None or not app.is_running():
                raise RuntimeError(
                    "[OVVisualizer] No Isaac Sim app is running. "
                    "OV visualizer requires Isaac Sim to be launched via AppLauncher before initialization. "
                    "Ensure your script calls AppLauncher before creating the environment."
                )
            
            # Try to get SimulationApp instance for headless check
            try:
                from isaacsim import SimulationApp
                
                # Check various ways SimulationApp might store its instance
                sim_app = None
                if hasattr(SimulationApp, '_instance') and SimulationApp._instance is not None:
                    sim_app = SimulationApp._instance
                elif hasattr(SimulationApp, 'instance') and callable(SimulationApp.instance):
                    sim_app = SimulationApp.instance()
                
                if sim_app is not None:
                    self._simulation_app = sim_app
                    
                    # Check if running in headless mode
                    if self._simulation_app.config.get("headless", False):
                        omni.log.warn(
                            "[OVVisualizer] Running in headless mode. "
                            "OV visualizer requires GUI mode (launch with --headless=False) to create viewports."
                        )
                    else:
                        omni.log.info("[OVVisualizer] Using existing Isaac Sim app instance.")
                else:
                    # App is running but we couldn't get SimulationApp instance
                    # This is okay - we can still use omni APIs
                    omni.log.info("[OVVisualizer] Isaac Sim app is running (via omni.kit.app).")
                    
            except ImportError:
                # SimulationApp not available, but omni.kit.app is running
                omni.log.info("[OVVisualizer] Using running Isaac Sim app (SimulationApp module not available).")
            
        except ImportError as e:
            raise ImportError(
                f"[OVVisualizer] Could not import omni.kit.app: {e}. "
                "Isaac Sim may not be installed or not running."
            )
    
    def _setup_viewport(self, usd_stage, metadata: dict) -> None:
        """Setup viewport with camera and window size."""
        try:
            import omni.kit.viewport.utility as vp_utils
            from omni.ui import DockPosition
            
            # Create new viewport or use existing
            if self.cfg.create_viewport and self.cfg.viewport_name:
                # Map dock position string to enum
                dock_position_map = {
                    "LEFT": DockPosition.LEFT,
                    "RIGHT": DockPosition.RIGHT,
                    "BOTTOM": DockPosition.BOTTOM,
                    "SAME": DockPosition.SAME,
                }
                dock_pos = dock_position_map.get(self.cfg.dock_position.upper(), DockPosition.SAME)
                
                # Create new viewport with proper API
                self._viewport_window = vp_utils.create_viewport_window(
                    name=self.cfg.viewport_name,
                    width=self.cfg.window_width,
                    height=self.cfg.window_height,
                    position_x=50,
                    position_y=50,
                    docked=True,
                )
                
                omni.log.info(f"[OVVisualizer] Created viewport '{self.cfg.viewport_name}'")
                
                # Dock the viewport asynchronously (needs to wait for window creation)
                asyncio.ensure_future(self._dock_viewport_async(self.cfg.viewport_name, dock_pos))
                
                # Create dedicated camera for this viewport
                if self._viewport_window:
                    self._create_and_assign_camera(usd_stage)
            else:
                # Use existing viewport by name, or fall back to active viewport
                if self.cfg.viewport_name:
                    self._viewport_window = vp_utils.get_viewport_window_by_name(self.cfg.viewport_name)
                    
                    if self._viewport_window is None:
                        omni.log.warn(
                            f"[OVVisualizer] Viewport '{self.cfg.viewport_name}' not found. "
                            f"Using active viewport instead."
                        )
                        self._viewport_window = vp_utils.get_active_viewport_window()
                    else:
                        omni.log.info(f"[OVVisualizer] Using existing viewport '{self.cfg.viewport_name}'")
                else:
                    self._viewport_window = vp_utils.get_active_viewport_window()
                    omni.log.info("[OVVisualizer] Using existing active viewport")
            
            if self._viewport_window is None:
                omni.log.warn("[OVVisualizer] Could not get/create viewport.")
                return
            
            # Get viewport API for camera control
            self._viewport_api = self._viewport_window.viewport_api
            
            # Set camera pose (uses existing camera if not created above)
            self._set_viewport_camera(self.cfg.camera_position, self.cfg.camera_target)
            
            omni.log.info(f"[OVVisualizer] Viewport configured (size: {self.cfg.window_width}x{self.cfg.window_height})")
            
        except ImportError as e:
            omni.log.warn(f"[OVVisualizer] Viewport utilities unavailable: {e}")
        except Exception as e:
            omni.log.error(f"[OVVisualizer] Error setting up viewport: {e}")
    
    async def _dock_viewport_async(self, viewport_name: str, dock_position) -> None:
        """Dock viewport window asynchronously after it's created.
        
        Args:
            viewport_name: Name of the viewport window to dock.
            dock_position: DockPosition enum value for where to dock.
        """
        try:
            import omni.ui
            import omni.kit.app
            
            # Wait for the viewport window to be created in the workspace
            viewport_window = None
            for i in range(10):  # Try up to 10 frames
                viewport_window = omni.ui.Workspace.get_window(viewport_name)
                if viewport_window:
                    omni.log.info(f"[OVVisualizer] Found viewport window '{viewport_name}' after {i} frames")
                    break
                await omni.kit.app.get_app().next_update_async()
            
            if not viewport_window:
                omni.log.warn(f"[OVVisualizer] Could not find viewport window '{viewport_name}' in workspace for docking.")
                return
            
            # Get the main viewport to dock relative to
            main_viewport = omni.ui.Workspace.get_window("Viewport")
            if not main_viewport:
                # Try alternative viewport names
                for alt_name in ["/OmniverseKit/Viewport", "Viewport Next"]:
                    main_viewport = omni.ui.Workspace.get_window(alt_name)
                    if main_viewport:
                        break
            
            if main_viewport and main_viewport != viewport_window:
                # Dock the new viewport relative to the main viewport
                viewport_window.dock_in(main_viewport, dock_position, 0.5)
                
                # Wait a frame for docking to complete
                await omni.kit.app.get_app().next_update_async()
                
                # Make the new viewport the active/focused tab
                # Try multiple methods to ensure it becomes active
                viewport_window.focus()
                viewport_window.visible = True
                
                # Wait another frame and focus again (sometimes needed for tabs)
                await omni.kit.app.get_app().next_update_async()
                viewport_window.focus()
                
                omni.log.info(f"[OVVisualizer] Docked viewport '{viewport_name}' at position {self.cfg.dock_position} and set as active")
            else:
                omni.log.info(f"[OVVisualizer] Could not find main viewport for docking. Viewport '{viewport_name}' will remain floating.")
                
        except Exception as e:
            omni.log.warn(f"[OVVisualizer] Error docking viewport: {e}")
    
    def _create_and_assign_camera(self, usd_stage) -> None:
        """Create a dedicated camera for this viewport and assign it."""
        try:
            from pxr import UsdGeom, Gf
            
            # Create camera prim path based on viewport name
            camera_path = f"/World/Cameras/{self.cfg.viewport_name}_Camera"
            
            # Check if camera already exists
            camera_prim = usd_stage.GetPrimAtPath(camera_path)
            if not camera_prim.IsValid():
                # Create camera prim
                camera = UsdGeom.Camera.Define(usd_stage, camera_path)
                omni.log.info(f"[OVVisualizer] Created camera: {camera_path}")
            else:
                omni.log.info(f"[OVVisualizer] Using existing camera: {camera_path}")
            
            # Assign camera to viewport
            if self._viewport_api:
                self._viewport_api.set_active_camera(camera_path)
                omni.log.info(f"[OVVisualizer] Assigned camera '{camera_path}' to viewport '{self.cfg.viewport_name}'")
            
        except Exception as e:
            omni.log.warn(f"[OVVisualizer] Could not create/assign camera: {e}. Using default camera.")
    
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
