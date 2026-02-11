# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse visualizer for Isaac Lab simulation contexts."""

from __future__ import annotations

import enum
import logging
import os
from typing import TYPE_CHECKING, Any

import flatdict
import toml

import omni.kit.app

from isaaclab.utils.version import get_isaac_sim_version
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
        local_gui = self._sim.get_setting("/app/window/enabled")
        livestream_gui = self._sim.get_setting("/app/livestream/enabled")
        xr_gui = self._sim.get_setting("/app/xr/enabled")
        self._offscreen_render = bool(self._sim.get_setting("/isaaclab/render/offscreen"))
        self._render_viewport = bool(self._sim.get_setting("/isaaclab/render/active_viewport"))

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

        # Apply render settings from RenderCfg (including preset loading)
        self._apply_render_settings_from_cfg()

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

        settings = self._sim._carb_settings

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

    def _apply_render_settings_from_cfg(self) -> None:  # noqa: C901
        """Apply RTX settings from RenderCfg, including loading preset files.

        This method applies rendering settings in the following order:
        1. Load and apply preset settings from rendering mode (performance/balanced/quality)
        2. Apply user-friendly named settings from RenderCfg
        3. Apply arbitrary carb settings from RenderCfg.carb_settings
        4. Apply antialiasing mode if specified

        # THINK(Octi): what if there is no visualizer but rtx camera, this should still be set, should this code
        # be add to visualizer at all? We should consider removing this code from visualizer and and to renderer
        # implementation once we see how renderer will handle this.
        """
        if self._sim is None:
            return

        render_cfg = getattr(self._sim.cfg, "render", None)
        if render_cfg is None:
            return

        # Define mapping of user-friendly RenderCfg names to native carb names
        rendering_setting_name_mapping = {
            "enable_translucency": "/rtx/translucency/enabled",
            "enable_reflections": "/rtx/reflections/enabled",
            "enable_global_illumination": "/rtx/indirectDiffuse/enabled",
            "enable_dlssg": "/rtx-transient/dlssg/enabled",
            "enable_dl_denoiser": "/rtx-transient/dldenoiser/enabled",
            "dlss_mode": "/rtx/post/dlss/execMode",
            "enable_direct_lighting": "/rtx/directLighting/enabled",
            "samples_per_pixel": "/rtx/directLighting/sampledLighting/samplesPerPixel",
            "enable_shadows": "/rtx/shadows/enabled",
            "enable_ambient_occlusion": "/rtx/ambientOcclusion/enabled",
            "dome_light_upper_lower_strategy": "/rtx/domeLight/upperLowerStrategy",
            "ambient_light_intensity": "/rtx/sceneDb/ambientLightIntensity",
            "ambient_occlusion_denoiser_mode": "/rtx/ambientOcclusion/denoiserMode",
            "subpixel_mode": "/rtx/raytracing/subpixel/mode",
            "enable_cached_raytracing": "/rtx/raytracing/cached/enabled",
            "max_samples_per_launch": "/rtx/pathtracing/maxSamplesPerLaunch",
            "view_tile_limit": "/rtx/viewTile/limit",
            # RT2 settings
            "max_bounces": "/rtx/rtpt/maxBounces",
            "split_glass": "/rtx/rtpt/splitGlass",
            "split_clearcoat": "/rtx/rtpt/splitClearcoat",
            "split_rough_reflection": "/rtx/rtpt/splitRoughReflection",
        }

        not_carb_settings = ["rendering_mode", "carb_settings", "antialiasing_mode"]

        # Grab the rendering mode using the following priority:
        # 1. Command line argument --rendering_mode, if provided
        # 2. rendering_mode from RenderCfg, if set
        # 3. Default to "balanced" mode, if neither is specified
        rendering_mode = self._sim.get_setting("/isaaclab/rendering/rendering_mode")
        if not rendering_mode:
            rendering_mode = getattr(render_cfg, "rendering_mode", None)
        if not rendering_mode:
            rendering_mode = "balanced"

        # Set preset settings (same behavior as the CLI arg --rendering_mode)
        if rendering_mode is not None:
            # Check if preset is supported
            supported_rendering_modes = ["performance", "balanced", "quality"]
            if rendering_mode not in supported_rendering_modes:
                raise ValueError(
                    f"RenderCfg rendering mode '{rendering_mode}' not in supported modes {supported_rendering_modes}."
                )

            # Grab Isaac Lab apps path
            isaaclab_app_exp_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), *[".."] * 5, "isaaclab", "apps"
            )
            # For Isaac Sim 5.0 compatibility, use the 5.X rendering mode app files in a different folder
            if get_isaac_sim_version().major < 6:
                isaaclab_app_exp_path = os.path.join(isaaclab_app_exp_path, "isaacsim_5")

            # Grab preset settings
            preset_filename = os.path.join(isaaclab_app_exp_path, f"rendering_modes/{rendering_mode}.kit")
            if os.path.exists(preset_filename):
                with open(preset_filename) as file:
                    preset_dict = toml.load(file)
                preset_dict = dict(flatdict.FlatDict(preset_dict, delimiter="."))

                # Set presets
                for key, value in preset_dict.items():
                    key = "/" + key.replace(".", "/")  # Convert to carb setting format
                    self._sim.set_setting(key, value)
            else:
                logger.warning(f"[OVVisualizer] Preset file not found: {preset_filename}")

        # Set user-friendly named settings
        for key, value in vars(render_cfg).items():
            if value is None or key in not_carb_settings:
                # Skip unset settings and non-carb settings
                continue
            if key not in rendering_setting_name_mapping:
                # Skip unknown keys (may be custom fields)
                continue
            carb_key = rendering_setting_name_mapping[key]
            self._sim.set_setting(carb_key, value)

        # Set general carb settings
        carb_settings = getattr(render_cfg, "carb_settings", None)
        if carb_settings is not None:
            for key, value in carb_settings.items():
                if "_" in key:
                    key = "/" + key.replace("_", "/")  # Convert from python variable style string
                elif "." in key:
                    key = "/" + key.replace(".", "/")  # Convert from .kit file style string
                if self._sim.get_setting(key) is None:
                    raise ValueError(f"'{key}' in RenderCfg.carb_settings does not map to a carb setting.")
                self._sim.set_setting(key, value)

        # Set antialiasing mode
        antialiasing_mode = getattr(render_cfg, "antialiasing_mode", None)
        if antialiasing_mode is not None:
            try:
                import omni.replicator.core as rep

                rep.settings.set_render_rtx_realtime(antialiasing=antialiasing_mode)
            except Exception:
                pass

        # WAR: Ensure /rtx/renderMode RaytracedLighting is correctly cased
        rendermode = self._sim.get_setting("/rtx/rendermode")
        if rendermode is not None and rendermode.lower() == "raytracedlighting":
            self._sim.set_setting("/rtx/rendermode", "RaytracedLighting")
            self._sim.set_setting("/rtx/pathtracing/lightcache/cached/alwaysReuse", True)

    def close(self) -> None:
        """Clean up visualizer resources."""
        self._sim = None
        self._app_iface = None
        self._viewport_context = None
        self._viewport_window = None
        self._is_initialized = False
        self._is_closed = True
