# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse visualizer for PhysX-based SimulationContext."""

from __future__ import annotations

import asyncio
import logging
import enum
from typing import Any
import omni.kit.app
from pxr import UsdGeom

from .ov_visualizer_cfg import OVVisualizerCfg
from .visualizer import Visualizer

logger = logging.getLogger(__name__)


class RenderMode(enum.IntEnum):
    """Different rendering modes for the simulation.

    Render modes correspond to how the viewport and other UI elements (such as listeners to keyboard or mouse
    events) are updated. There are three main components that can be updated when the simulation is rendered:

    1. **UI elements and other extensions**: These are UI elements (such as buttons, sliders, etc.) and other
       extensions that are running in the background that need to be updated when the simulation is running.
    2. **Cameras**: These are typically based on Hydra textures and are used to render the scene from different
       viewpoints. They can be attached to a viewport or be used independently to render the scene.
    3. **Viewports**: These are windows where you can see the rendered scene.

    Updating each of the above components has a different overhead. For example, updating the viewports is
    computationally expensive compared to updating the UI elements. Therefore, it is useful to be able to
    control what is updated when the simulation is rendered. This is where the render mode comes in. There are
    four different render modes:

    * :attr:`NO_GUI_OR_RENDERING`: The simulation is running without a GUI and off-screen rendering flag is disabled,
      so none of the above are updated.
    * :attr:`NO_RENDERING`: No rendering, where only 1 is updated at a lower rate.
    * :attr:`PARTIAL_RENDERING`: Partial rendering, where only 1 and 2 are updated.
    * :attr:`FULL_RENDERING`: Full rendering, where everything (1, 2, 3) is updated.

    .. _Viewports: https://docs.omniverse.nvidia.com/extensions/latest/ext_viewport.html
    """

    NO_GUI_OR_RENDERING = -1
    """The simulation is running without a GUI and off-screen rendering is disabled."""
    NO_RENDERING = 0
    """No rendering, where only other UI elements are updated at a lower rate."""
    PARTIAL_RENDERING = 1
    """Partial rendering, where the simulation cameras and UI elements are updated."""
    FULL_RENDERING = 2
    """Full rendering, where all the simulation viewports, cameras and UI elements are updated."""


class OVVisualizer(Visualizer):
    """Omniverse visualizer managing viewport/rendering.

    This class extends the base :class:`Visualizer` and handles:
    - Viewport context and window management
    - Render mode switching
    - Camera view setup
    - Render settings from configuration
    - Throttled rendering for UI responsiveness

    Lifecycle: __init__(cfg) -> initialize(scene_data) -> step() (repeated) -> close()
    """

    def __init__(self, cfg: "OVVisualizerCfg"):
        """Initialize OV visualizer with configuration.

        Args:
            cfg: Configuration for the visualizer.
        """
        super().__init__(cfg)
        self.cfg: "OVVisualizerCfg" = cfg

        # Will be set during initialize()
        self._simulation_context = None
        self._simulation_app = None
        self._simulation_app_running = False

        # Viewport state
        self._viewport_context = None
        self._viewport_window = None
        self._viewport_api = None
        self._render_throttle_counter = 0
        self._render_throttle_period = cfg.render_throttle_period

        # Render mode
        self.render_mode = RenderMode.NO_GUI_OR_RENDERING

    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        """Initialize visualizer with simulation context.

        Args:
            scene_data: Dictionary containing:
                - 'simulation_context': The SimulationContext instance (required)
                - 'usd_stage': The USD stage (optional, can get from sim context)
        """
        if self._is_initialized:
            logger.warning("[OVVisualizer] Already initialized.")
            return

        usd_stage = None
        if scene_data is not None:
            usd_stage = scene_data.get("usd_stage")
            self._simulation_context = scene_data.get("simulation_context")

        if usd_stage is None:
            raise RuntimeError("OV visualizer requires a USD stage.")

        # Build metadata from simulation context if available
        metadata = {}
        if self._simulation_context is not None:
            # Try to get num_envs from the simulation context's scene if available
            num_envs = 0
            if hasattr(self._simulation_context, "scene") and self._simulation_context.scene is not None:
                if hasattr(self._simulation_context.scene, "num_envs"):
                    num_envs = self._simulation_context.scene.num_envs

            # Detect physics backend (could be extended to check actual backend type)
            physics_backend = "newton"  # Default for now, could be made more sophisticated

            metadata = {
                "num_envs": num_envs,
                "physics_backend": physics_backend,
                "env_prim_pattern": "/World/envs/env_{}",  # Standard pattern
            }

            self._simulation_context.settings.set_bool("/app/player/playSimulations", False)
            self._offscreen_render = bool(self._simulation_context.settings.get("/isaaclab/render/offscreen"))
            self._render_viewport = bool(self._simulation_context.settings.get("/isaaclab/render/active_viewport"))
            self._has_gui = bool(self._simulation_context.settings.get("/isaaclab/render/active_viewport"))
            # Set render mode
            if not self._has_gui and not self._offscreen_render:
                self.render_mode = RenderMode.NO_GUI_OR_RENDERING
            elif not self._has_gui and self._offscreen_render:
                self.render_mode = RenderMode.PARTIAL_RENDERING
            else:
                self.render_mode = RenderMode.FULL_RENDERING

            # enable viewport updates if GUI is enabled
            if self._has_gui:
                try:
                    import omni.ui as ui
                    from omni.kit.viewport.utility import get_active_viewport
                    self._viewport_context = get_active_viewport()
                    self._viewport_context.updates_enabled = True
                    self._viewport_window = ui.Workspace.get_window("Viewport")
                except (ImportError, AttributeError):
                    pass

            # Disable viewport for offscreen-only rendering
            if not self._render_viewport and self._offscreen_render:
                try:
                    from omni.kit.viewport.utility import get_active_viewport
                    get_active_viewport().updates_enabled = False
                except (ImportError, AttributeError):
                    pass

        self._ensure_simulation_app()
        self._setup_viewport(usd_stage, metadata)

        num_envs = metadata.get("num_envs", 0)
        physics_backend = metadata.get("physics_backend", "unknown")
        logger.info(f"[OVVisualizer] Initialized ({num_envs} envs, {physics_backend} physics)")

        self._is_initialized = True

    def step(self, dt: float, state: Any | None = None) -> None:
        """Update visualization for one step (render the scene).

        Args:
            dt: Time step in seconds.
            state: Updated physics state (unused for OV - USD stage is auto-synced).
        """
        if not self._is_initialized:
            return

        self.render()

    def render(self, mode: RenderMode | None = None) -> None:
        """Refreshes the rendering components including UI elements and view-ports depending on the render mode.

        This function is used to refresh the rendering components of the simulation. This includes updating the
        view-ports, UI elements, and other extensions (besides physics simulation) that are running in the
        background. The rendering components are refreshed based on the render mode.

        Please see :class:`RenderMode` for more information on the different render modes.

        Args:
            mode: The rendering mode. Defaults to None, in which case the current rendering mode is used.
        """
        if self._simulation_context is None:
            return

        # check if we need to change the render mode
        if mode is not None:
            self.set_render_mode(mode)

        self._simulation_context.settings.set_bool("/app/player/playSimulations", False)
        omni.kit.app.get_app().update()
        self._simulation_context.settings.set_bool("/app/player/playSimulations", True)

        # app.update() may be changing the cuda device, so we force it back to our desired device here
        if "cuda" in self._simulation_context.device:
            import torch

            torch.cuda.set_device(self._simulation_context.device)

    def is_running(self) -> bool:
        """Check if visualizer is still running."""
        return self.is_playing()

    def is_playing(self) -> bool:
        """Check whether the simulation is playing."""
        if self._timeline is None:
            return False
        return self._timeline.is_playing()

    def is_stopped(self) -> bool:
        """Check whether the simulation is stopped."""
        if self._timeline is None:
            return True
        return self._timeline.is_stopped()

    def play(self) -> None:
        """Start playing the simulation."""
        if self._timeline is not None:
            self._timeline.play()

    def pause(self) -> None:
        """Pause the simulation."""
        if self._timeline is not None:
            self._timeline.pause()

    def stop(self) -> None:
        """Stop the simulation."""
        if self._timeline is not None:
            self._timeline.stop()

    def is_training_paused(self) -> bool:
        """Check if training is paused (always False for OV)."""
        return False

    def supports_markers(self) -> bool:
        """Supports markers via USD prims."""
        return True

    def supports_live_plots(self) -> bool:
        """Supports live plots via Isaac Lab UI."""
        return True

    def get_rendering_dt(self) -> float | None:
        """Get rendering dt based on OV rate limiting settings."""
        if self._simulation_context is None:
            return None

        settings = self._simulation_context.settings

        def _from_frequency():
            freq = settings.get("/app/runLoops/main/rateLimitFrequency")
            return 1.0 / freq if freq else None

        if settings.get("/app/runLoops/main/rateLimitEnabled"):
            return _from_frequency()

        try:
            import omni.kit.loop._loop as omni_loop

            runner = omni_loop.acquire_loop_interface()
            return runner.get_manual_step_size() if runner.get_manual_mode() else _from_frequency()
        except Exception:
            return _from_frequency()

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set the location and target of the viewport camera in the stage.

        Note:
            This is a wrapper around the :math:`isaacsim.core.utils.viewports.set_camera_view` function.
            It is provided here for convenience to reduce the amount of imports needed.

        Args:
            eye: The location of the camera eye.
            target: The location of the camera target.
            camera_prim_path: The path to the camera primitive in the stage. Defaults to
                the value from config or "/OmniverseKit_Persp".
        """
        if not self._is_initialized:
            logger.warning("[OVVisualizer] Cannot set camera view - visualizer not initialized.")
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
                eye=list(eye), target=list(target), camera_prim_path=camera_path, viewport_api=self._viewport_api
            )

            logger.info(f"[OVVisualizer] Camera set: pos={eye}, target={target}, camera={camera_path}")

        except Exception as e:
            logger.warning(f"[OVVisualizer] Could not set camera: {e}")

    def set_render_mode(self, mode: RenderMode) -> None:
        """Change the current render mode of the simulation.

        Please see :class:`RenderMode` for more information on the different render modes.

        .. note::
            When no GUI is available (locally or livestreamed), we do not need to choose whether the viewport
            needs to render or not (since there is no GUI). Thus, in this case, calling the function will not
            change the render mode.

        Args:
            mode: The rendering mode. If different than current rendering mode,
                the mode is changed to the new mode.

        Raises:
            ValueError: If the input mode is not supported.
        """
        # check if mode change is possible -- not possible when no GUI is available
        if not getattr(self, "_has_gui", True):
            logger.warning(
                f"Cannot change render mode when GUI is disabled. Using the default render mode: {self.render_mode}."
            )
            return
        # check if there is a mode change
        # note: this is mostly needed for GUI when we want to switch between full rendering and no rendering.
        if mode != self.render_mode:
            if mode == RenderMode.FULL_RENDERING:
                # display the viewport and enable updates
                self._viewport_context.updates_enabled = True  # pyright: ignore [reportOptionalMemberAccess]
                self._viewport_window.visible = True  # pyright: ignore [reportOptionalMemberAccess]
            elif mode == RenderMode.PARTIAL_RENDERING:
                # hide the viewport and disable updates
                self._viewport_context.updates_enabled = False  # pyright: ignore [reportOptionalMemberAccess]
                self._viewport_window.visible = False  # pyright: ignore [reportOptionalMemberAccess]
            elif mode == RenderMode.NO_RENDERING:
                # hide the viewport and disable updates
                if self._viewport_context is not None:
                    self._viewport_context.updates_enabled = False  # pyright: ignore [reportOptionalMemberAccess]
                    self._viewport_window.visible = False  # pyright: ignore [reportOptionalMemberAccess]
                # reset the throttle counter
                self._render_throttle_counter = 0
            else:
                raise ValueError(f"Unsupported render mode: {mode}! Please check `RenderMode` for details.")
            # update render mode
            self.render_mode = mode

    def close(self) -> None:
        """Clean up visualizer resources."""
        # Note: We don't close the SimulationApp here as it's managed by AppLauncher
        self._simulation_context = None
        self._simulation_app = None
        self._viewport_context = None
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False
        self._is_closed = True

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _ensure_simulation_app(self) -> None:
        """Ensure Isaac Sim app is running."""
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
                if hasattr(SimulationApp, "_instance") and SimulationApp._instance is not None:
                    sim_app = SimulationApp._instance
                elif hasattr(SimulationApp, "instance") and callable(SimulationApp.instance):
                    sim_app = SimulationApp.instance()

                if sim_app is not None:
                    self._simulation_app = sim_app

                    # Check if running in headless mode
                    if self._simulation_app.config.get("headless", False):  # pyright: ignore [reportAttributeAccessIssue]
                        logger.warning(
                            "[OVVisualizer] Running in headless mode. "
                            "OV visualizer requires GUI mode (launch with --headless=False) to create viewports."
                        )
                    else:
                        logger.info("[OVVisualizer] Using existing Isaac Sim app instance.")
                else:
                    # App is running but we couldn't get SimulationApp instance
                    # This is okay - we can still use omni APIs
                    logger.info("[OVVisualizer] Isaac Sim app is running (via omni.kit.app).")
                self._simulation_app_running = True
            except ImportError:
                # SimulationApp not available, but omni.kit.app is running
                logger.info("[OVVisualizer] Using running Isaac Sim app (SimulationApp module not available).")

        except ImportError as e:
            raise ImportError(
                f"[OVVisualizer] Could not import omni.kit.app: {e}. Isaac Sim may not be installed or not running."
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

                logger.info(f"[OVVisualizer] Created viewport '{self.cfg.viewport_name}'")

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
                        logger.warning(
                            f"[OVVisualizer] Viewport '{self.cfg.viewport_name}' not found. "
                            "Using active viewport instead."
                        )
                        self._viewport_window = vp_utils.get_active_viewport_window()
                    else:
                        logger.info(f"[OVVisualizer] Using existing viewport '{self.cfg.viewport_name}'")
                else:
                    self._viewport_window = vp_utils.get_active_viewport_window()
                    logger.info("[OVVisualizer] Using existing active viewport")

            if self._viewport_window is None:
                logger.warning("[OVVisualizer] Could not get/create viewport.")
                return

            # Get viewport API for camera control
            self._viewport_api = self._viewport_window.viewport_api

            # Set camera pose (uses existing camera if not created above)
            self.set_camera_view(self.cfg.camera_position, self.cfg.camera_target)

            logger.info(f"[OVVisualizer] Viewport configured (size: {self.cfg.window_width}x{self.cfg.window_height})")

        except ImportError as e:
            logger.warning(f"[OVVisualizer] Viewport utilities unavailable: {e}")
        except Exception as e:
            logger.error(f"[OVVisualizer] Error setting up viewport: {e}")

    async def _dock_viewport_async(self, viewport_name: str, dock_position) -> None:
        """Dock viewport window asynchronously after it's created.

        Args:
            viewport_name: Name of the viewport window to dock.
            dock_position: DockPosition enum value for where to dock.
        """
        try:
            import omni.kit.app
            import omni.ui

            # Wait for the viewport window to be created in the workspace
            viewport_window = None
            for i in range(10):  # Try up to 10 frames
                viewport_window = omni.ui.Workspace.get_window(viewport_name)
                if viewport_window:
                    logger.info(f"[OVVisualizer] Found viewport window '{viewport_name}' after {i} frames")
                    break
                await omni.kit.app.get_app().next_update_async()

            if not viewport_window:
                logger.warning(
                    f"[OVVisualizer] Could not find viewport window '{viewport_name}' in workspace for docking."
                )
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

                logger.info(
                    f"[OVVisualizer] Docked viewport '{viewport_name}' at position {self.cfg.dock_position} and set as"
                    " active"
                )
            else:
                logger.info(
                    f"[OVVisualizer] Could not find main viewport for docking. Viewport '{viewport_name}' will remain"
                    " floating."
                )

        except Exception as e:
            logger.warning(f"[OVVisualizer] Error docking viewport: {e}")

    def _create_and_assign_camera(self, usd_stage) -> None:
        """Create a dedicated camera for this viewport and assign it."""
        try:
            # Create camera prim path based on viewport name
            camera_path = f"/World/Cameras/{self.cfg.viewport_name}_Camera"

            # Check if camera already exists
            camera_prim = usd_stage.GetPrimAtPath(camera_path)
            if not camera_prim.IsValid():
                # Create camera prim
                UsdGeom.Camera.Define(usd_stage, camera_path)
                logger.info(f"[OVVisualizer] Created camera: {camera_path}")
            else:
                logger.info(f"[OVVisualizer] Using existing camera: {camera_path}")

            # Assign camera to viewport
            if self._viewport_api:
                self._viewport_api.set_active_camera(camera_path)
                logger.info(f"[OVVisualizer] Assigned camera '{camera_path}' to viewport '{self.cfg.viewport_name}'")

        except Exception as e:
            logger.warning(f"[OVVisualizer] Could not create/assign camera: {e}. Using default camera.")
