# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse visualizer for Isaac Lab simulation contexts."""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Any

import omni.kit.app

from isaaclab.visualizers import Visualizer

from .ov_visualizer_cfg import OVVisualizerCfg

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class RenderMode(enum.IntEnum):
    """Different rendering modes for the simulation.

    Render modes correspond to how the viewport and other UI elements
    (such as listeners to keyboard or mouse events) are updated.
    There are three main components that can be updated when the
    simulation is rendered:

    1. **UI elements and other extensions**: These are UI elements
       (such as buttons, sliders, etc.) and other extensions that are
       running in the background that need to be updated when the
       simulation is running.
    2. **Cameras**: These are typically based on Hydra textures and are
       used to render the scene from different viewpoints. They can be
       attached to a viewport or be used independently to render the
       scene.
    3. **Viewports**: These are windows where you can see the rendered
       scene.

    Updating each of the above components has a different overhead. For
    example, updating the viewports is computationally expensive
    compared to updating the UI elements. Therefore, it is useful to be
    able to control what is updated when the simulation is rendered.
    This is where the render mode comes in. There are four different
    render modes:

    * :attr:`NO_GUI_OR_RENDERING`: The simulation is running without a
      GUI and off-screen rendering flag is disabled, so none of the
      above are updated.
    * :attr:`NO_RENDERING`: No rendering, where only 1 is updated at a
      lower rate.
    * :attr:`PARTIAL_RENDERING`: Partial rendering, where only 1 and 2
      are updated.
    * :attr:`FULL_RENDERING`: Full rendering, where everything (1, 2,
      3) is updated.

    .. _Viewports: https://docs.omniverse.nvidia.com/extensions/latest
       /ext_viewport.html
    """

    NO_GUI_OR_RENDERING = -1
    """The simulation runs without a GUI and off-screen rendering."""
    NO_RENDERING = 0
    """No rendering, where only other UI elements are updated."""
    PARTIAL_RENDERING = 1
    """Partial rendering, where cameras and UI elements are updated."""
    FULL_RENDERING = 2
    """Full rendering, where viewports, cameras and UI are updated."""


class OVVisualizer(Visualizer):
    """Omniverse visualizer managing viewport/rendering.

    This class extends the base :class:`Visualizer` and handles:
    - Viewport context and window management
    - Render mode switching
    - Camera view setup
    - Render settings from configuration

    Lifecycle:
        __init__(cfg) -> initialize(scene_data) -> step() -> close()
    """

    def __init__(self, cfg: OVVisualizerCfg):
        """Initialize OV visualizer with configuration.

        Args:
            cfg: Configuration for the visualizer.
        """
        super().__init__(cfg)
        self.cfg: OVVisualizerCfg = cfg

        # Will be set during initialize()
        self._sim: SimulationContext | None = None
        self._app_iface: omni.kit.app.IApp | None = None

        # Render state
        self._has_gui = False
        self._offscreen_render = False
        self._render_viewport = False

        # Viewport state
        self._viewport_context = None
        self._viewport_window = None
        self._render_throttle_counter = 0
        self._render_throttle_period = cfg.render_throttle_period

        # Render mode
        self.render_mode = RenderMode.NO_GUI_OR_RENDERING

    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        """Initialize visualizer with simulation context.

        Args:
            scene_data: Dictionary containing:
                - 'simulation_context': SimulationContext instance
        """
        if self._is_initialized:
            logger.warning("[OVVisualizer] Already initialized.")
            return

        if scene_data is None:
            raise ValueError("OVVisualizer requires scene_data with 'simulation_context'")

        self._sim = scene_data.get("simulation_context")
        if self._sim is None:
            raise ValueError("OVVisualizer requires 'simulation_context' in scene_data")

        # Acquire application interface
        self._app_iface = omni.kit.app.get_app_interface()

        # Detect render flags from carb settings
        local_gui = self._sim.carb_settings.get("/app/window/enabled")
        livestream_gui = self._sim.carb_settings.get("/app/livestream/enabled")
        xr_gui = self._sim.carb_settings.get("/app/xr/enabled")
        self._offscreen_render = bool(self._sim.carb_settings.get("/isaaclab/render/offscreen"))
        self._render_viewport = bool(self._sim.carb_settings.get("/isaaclab/render/active_viewport"))

        # Flag for whether any GUI will be rendered
        self._has_gui = bool(local_gui or livestream_gui or xr_gui)
        self._sim.set_setting("/isaaclab/has_gui", self._has_gui)

        # Set render mode based on GUI/offscreen settings
        if self.cfg.default_render_mode is not None:
            self.render_mode = self.cfg.default_render_mode
        elif not self._has_gui and not self._offscreen_render:
            self.render_mode = RenderMode.NO_GUI_OR_RENDERING
        elif not self._has_gui and self._offscreen_render:
            self.render_mode = RenderMode.PARTIAL_RENDERING
        else:
            import omni.ui as ui
            from omni.kit.viewport.utility import get_active_viewport

            self.render_mode = RenderMode.FULL_RENDERING
            self._viewport_context = get_active_viewport()
            self._viewport_context.updates_enabled = True
            viewport_name = self.cfg.viewport_name or "Viewport"
            self._viewport_window = ui.Workspace.get_window(viewport_name)

        # Disable viewport for offscreen-only rendering
        if not self._render_viewport and self._offscreen_render:
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().updates_enabled = False

        # Override enable scene querying if GUI is enabled
        if self._has_gui:
            self._sim.cfg.enable_scene_query_support = True

        # Set initial camera view
        self.set_camera_view(self.cfg.camera_position, self.cfg.camera_target)

        self._is_initialized = True
        logger.info("[OVVisualizer] Initialized")

    def step(self, dt: float, state: Any | None = None) -> None:
        """Update visualization for one step (render the scene).

        Args:
            dt: Time step in seconds.
            state: Updated physics state (unused - USD stage is synced).
        """
        if not self._is_initialized:
            return

        self.render()

    def render(self, mode: RenderMode | None = None) -> None:
        """Refreshes rendering components based on the render mode.

        This function is used to refresh the rendering components of
        the simulation. This includes updating the view-ports, UI
        elements, and other extensions (besides physics simulation)
        that are running in the background. The rendering components
        are refreshed based on the render mode.

        Please see :class:`RenderMode` for more information on the
        different render modes.

        Args:
            mode: The rendering mode. Defaults to None, in which case
                the current rendering mode is used.
        """
        if self._sim is None or self._app_iface is None:
            return

        # Check if we need to change the render mode
        if mode is not None:
            self.set_render_mode(mode)

        # Update the app (render)
        self._sim.set_setting("/app/player/playSimulations", False)
        self._app_iface.update()
        self._sim.set_setting("/app/player/playSimulations", True)

        # app.update() may change the cuda device, so force it back
        if "cuda" in self._sim.device:
            import torch

            torch.cuda.set_device(self._sim.device)

    @property
    def has_gui(self) -> bool:
        """Whether any GUI is available (local, livestreamed, or XR)."""
        return self._has_gui

    @property
    def app(self) -> omni.kit.app.IApp | None:
        """Omniverse Kit Application interface."""
        return self._app_iface

    @property
    def offscreen_render(self) -> bool:
        """Whether offscreen rendering is enabled."""
        return self._offscreen_render

    @property
    def render_viewport(self) -> bool:
        """Whether the default viewport should be rendered."""
        return self._render_viewport

    def is_running(self) -> bool:
        """Check if visualizer is still running."""
        return self._is_initialized and not self._is_closed

    def is_stopped(self) -> bool:
        """Check if visualizer is stopped (closed)."""
        return self._is_closed

    def supports_markers(self) -> bool:
        """Supports markers via USD prims."""
        return True

    def supports_live_plots(self) -> bool:
        """Supports live plots via Isaac Lab UI."""
        return True

    def get_rendering_dt(self) -> float | None:
        """Get rendering dt based on OV rate limiting settings."""
        if self._sim is None:
            return None

        settings = self._sim.carb_settings

        def _from_frequency():
            freq = settings.get("/app/runLoops/main/rateLimitFrequency")
            return 1.0 / freq if freq else None

        if settings.get("/app/runLoops/main/rateLimitEnabled"):
            return _from_frequency()

        try:
            import omni.kit.loop._loop as omni_loop

            runner = omni_loop.acquire_loop_interface()
            if runner.get_manual_mode():
                return runner.get_manual_step_size()
            return _from_frequency()
        except Exception:
            return _from_frequency()

    def set_camera_view(
        self,
        eye: tuple[float, float, float] | list[float],
        target: tuple[float, float, float] | list[float],
    ) -> None:
        """Set the location and target of the viewport camera.

        Args:
            eye: The location of the camera eye.
            target: The location of the camera target.
        """
        if not self._is_initialized:
            logger.warning("[OVVisualizer] Cannot set camera - not initialized.")
            return

        try:
            import isaacsim.core.utils.viewports as vp_utils

            camera_path = self.cfg.camera_prim_path
            vp_utils.set_camera_view(
                eye=list(eye),
                target=list(target),
                camera_prim_path=camera_path,
            )
            logger.info(f"[OVVisualizer] Camera: pos={eye}, target={target}")
        except Exception as e:
            logger.warning(f"[OVVisualizer] Could not set camera: {e}")

    def set_render_mode(self, mode: RenderMode) -> None:
        """Change the current render mode of the simulation.

        Please see :class:`RenderMode` for more information on the
        different render modes.

        Args:
            mode: The rendering mode.

        Raises:
            ValueError: If the input mode is not supported.
        """
        # Check if mode change is possible -- not possible when no GUI
        if not self._has_gui:
            logger.warning(
                f"Cannot change render mode when GUI is disabled. Using the default render mode: {self.render_mode}."
            )
            return

        # Check if there is a mode change
        if mode != self.render_mode:
            if mode == RenderMode.FULL_RENDERING:
                if self._viewport_context is not None:
                    self._viewport_context.updates_enabled = True
                if self._viewport_window is not None:
                    self._viewport_window.visible = True
            elif mode == RenderMode.PARTIAL_RENDERING:
                if self._viewport_context is not None:
                    self._viewport_context.updates_enabled = False
                if self._viewport_window is not None:
                    self._viewport_window.visible = False
            elif mode == RenderMode.NO_RENDERING:
                if self._viewport_context is not None:
                    self._viewport_context.updates_enabled = False
                if self._viewport_window is not None:
                    self._viewport_window.visible = False
                self._render_throttle_counter = 0
            else:
                raise ValueError(f"Unsupported render mode: {mode}! Please check `RenderMode` for details.")
            self.render_mode = mode

    def close(self) -> None:
        """Clean up visualizer resources."""
        self._sim = None
        self._app_iface = None
        self._viewport_context = None
        self._viewport_window = None
        self._is_initialized = False
        self._is_closed = True
