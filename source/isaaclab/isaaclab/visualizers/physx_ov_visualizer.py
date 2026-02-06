# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse visualizer for PhysX-based SimulationContext."""

from __future__ import annotations

import enum
import flatdict
import logging
import os
import toml
import torch
import weakref
from typing import TYPE_CHECKING, Any, Callable

import omni.kit.app
import omni.timeline
from isaaclab.utils.version import get_isaac_sim_version
from .visualizer import Visualizer

if TYPE_CHECKING:
    import carb

    from .physx_ov_visualizer_cfg import PhysxOVVisualizerCfg
    from isaaclab.sim.simulation_context import SimulationContext

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


class TimelineControl:
    """Helper class for managing timeline lifecycle (play/pause/stop).

    This class wraps the omni.timeline interface and provides a clean API
    for controlling simulation playback. It can be composed by visualizers
    or simulation contexts that need timeline control.

    Features:
        - Play/pause/stop control with proper physics handle propagation
        - Stop event callback subscription
        - Timeline state queries (is_playing, is_stopped)
    """

    def __init__(
        self,
        app_interface: omni.kit.app.IApp,
        carb_settings: "carb.settings.ISettings",
        on_stop_callback: Callable[[], None] | None = None,
    ):
        """Initialize timeline control.

        Args:
            app_interface: Omniverse Kit application interface.
            carb_settings: Carb settings interface for controlling playSimulations.
            on_stop_callback: Optional callback to invoke when simulation stops.
        """
        self._app_iface = app_interface
        self._carb_settings = carb_settings
        self._on_stop_callback = on_stop_callback

        # Acquire timeline interface
        self._timeline_iface = omni.timeline.get_timeline_interface()
        self._timeline_iface.set_auto_update(True)

        # Setup stop handle callback
        self._disable_stop_handle = False
        timeline_event_stream = self._timeline_iface.get_timeline_event_stream()
        self._stop_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda *args, obj=weakref.proxy(self): obj._on_stop_event(*args),
            order=15,
        )

    @property
    def timeline_interface(self) -> omni.timeline.ITimeline:
        """Get the underlying timeline interface."""
        return self._timeline_iface

    def is_playing(self) -> bool:
        """Check whether the simulation is playing.

        Returns:
            True if simulation is currently playing, False otherwise.
        """
        return self._timeline_iface.is_playing()

    def is_stopped(self) -> bool:
        """Check whether the simulation is stopped.

        Returns:
            True if simulation is stopped, False otherwise.
        """
        return self._timeline_iface.is_stopped()

    def play(self) -> None:
        """Start playing the simulation.

        This commits the timeline state and performs one app update step
        to propagate all physics handles properly.
        """
        self._timeline_iface.play()
        self._timeline_iface.commit()
        # Perform one step to propagate all physics handles properly
        self._carb_settings.set_bool("/app/player/playSimulations", False)
        self._app_iface.update()
        self._carb_settings.set_bool("/app/player/playSimulations", True)

    def pause(self) -> None:
        """Pause the simulation.

        This commits the timeline state to ensure the pause takes effect.
        """
        self._timeline_iface.pause()
        self._timeline_iface.commit()

    def stop(self) -> None:
        """Stop the simulation.

        This commits the timeline state and performs one app update step
        to propagate all physics handles properly.
        """
        self._timeline_iface.stop()
        self._timeline_iface.commit()
        # Perform one step to propagate all physics handles properly
        self._carb_settings.set_bool("/app/player/playSimulations", False)
        self._app_iface.update()
        self._carb_settings.set_bool("/app/player/playSimulations", True)

    def set_stop_handle_enabled(self, enabled: bool) -> None:
        """Enable or disable the stop handle callback.

        When disabled, the on_stop_callback will not be invoked when
        the simulation stops.

        Args:
            enabled: If True, the stop handle callback is active.
        """
        self._disable_stop_handle = not enabled

    def set_target_framerate(self, hz: int) -> None:
        """Set the target framerate for the timeline.

        Args:
            hz: Target framerate in Hz.
        """
        self._timeline_iface.set_target_framerate(hz)

    def set_time_codes_per_second(self, time_codes_per_second: float) -> None:
        """Set the time codes per second for the timeline.

        Args:
            time_codes_per_second: Number of time codes per second.
        """
        self._timeline_iface.set_time_codes_per_second(time_codes_per_second)

    def close(self) -> None:
        """Clean up timeline resources.

        Unsubscribes from the stop event and releases the timeline handle.
        """
        if self._stop_handle is not None:
            self._stop_handle.unsubscribe()
            self._stop_handle = None

    def _on_stop_event(self, _) -> None:
        """Internal callback when simulation stops.

        Invokes the on_stop_callback if the stop handle is enabled.
        """
        if not self._disable_stop_handle and self._on_stop_callback is not None:
            self._on_stop_callback()


class PhysxOVVisualizer(Visualizer):
    """Omniverse visualizer managing viewport/rendering for PhysX workflow.

    This class extends the base :class:`Visualizer` and handles:
    - Viewport context and window management
    - Render mode switching
    - Camera view setup
    - Render settings from configuration
    - Throttled rendering for UI responsiveness
    - Timeline control (play/pause/stop)

    Lifecycle: __init__(cfg) -> initialize(scene_data) -> step() (repeated) -> close()
    """

    def __init__(self, cfg: "PhysxOVVisualizerCfg"):
        """Initialize PhysX OV visualizer with configuration.

        Args:
            cfg: Configuration for the visualizer.
        """
        super().__init__(cfg)
        self.cfg: "PhysxOVVisualizerCfg" = cfg

        # Will be set during initialize()
        self._sim: "SimulationContext | None" = None
        self._app_iface = None
        self._timeline: TimelineControl | None = None

        # Render state
        self._local_gui = False
        self._livestream_gui = False
        self._xr_gui = False
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
                - 'simulation_context': The SimulationContext instance (required)
                - 'usd_stage': The USD stage (optional, can get from sim context)
        """
        if self._is_initialized:
            logger.warning("[PhysxOVVisualizer] Already initialized.")
            return

        if scene_data is None:
            raise ValueError("PhysxOVVisualizer requires scene_data with 'simulation_context'")

        self._sim = scene_data.get("simulation_context")
        if self._sim is None:
            raise ValueError("PhysxOVVisualizer requires 'simulation_context' in scene_data")

        # Acquire application interface
        self._app_iface = omni.kit.app.get_app_interface()

        # Detect render flags
        self._local_gui = self._sim.carb_settings.get("/app/window/enabled")
        self._livestream_gui = self._sim.carb_settings.get("/app/livestream/enabled")
        self._xr_gui = self._sim.carb_settings.get("/app/xr/enabled")
        self._offscreen_render = bool(self._sim.carb_settings.get("/isaaclab/render/offscreen"))
        self._render_viewport = bool(self._sim.carb_settings.get("/isaaclab/render/active_viewport"))

        # Flag for whether any GUI will be rendered (local, livestreamed or viewport)
        has_gui = self._local_gui or self._livestream_gui or self._xr_gui
        self._sim.carb_settings.set_bool("/isaaclab/has_gui", has_gui)

        # Apply render settings from config
        self._apply_render_settings_from_cfg()

        # Store the default render mode
        if self.cfg.default_render_mode is not None:
            self.render_mode = self.cfg.default_render_mode
        elif not has_gui and not self._offscreen_render:
            # set default render mode
            # note: this is the terminal state: cannot exit from this render mode
            self.render_mode = RenderMode.NO_GUI_OR_RENDERING
        elif not has_gui and self._offscreen_render:
            # set default render mode
            # note: this is the terminal state: cannot exit from this render mode
            self.render_mode = RenderMode.PARTIAL_RENDERING
        else:
            # note: need to import here in case the UI is not available (ex. headless mode)
            import omni.ui as ui
            from omni.kit.viewport.utility import get_active_viewport

            # set default render mode
            # note: this can be changed by calling the `set_render_mode` function
            self.render_mode = RenderMode.FULL_RENDERING
            # acquire viewport context
            self._viewport_context = get_active_viewport()
            self._viewport_context.updates_enabled = True  # pyright: ignore [reportOptionalMemberAccess]
            # acquire viewport window
            self._viewport_window = ui.Workspace.get_window(self.cfg.viewport_name or "Viewport")

        # Check the case where we don't need to render the viewport
        # since render_viewport can only be False in headless mode, we only need to check for offscreen_render
        if not self._render_viewport and self._offscreen_render:
            # disable the viewport if offscreen_render is enabled
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().updates_enabled = False

        # Override enable scene querying if rendering is enabled
        # this is needed for some GUI features
        if self._sim.carb_settings.get("/isaaclab/has_gui"):
            self._sim.cfg.enable_scene_query_support = True

        # Initialize timeline control (manages play/pause/stop and stop callbacks)
        self._timeline = TimelineControl(
            app_interface=self._app_iface,
            carb_settings=self._sim.carb_settings,
            on_stop_callback=self._on_timeline_stop,
        )

        # Configure rendering/timeline rate
        self._configure_rendering_dt()

        # Set initial camera view
        self.set_camera_view(self.cfg.camera_position, self.cfg.camera_target)

        self._is_initialized = True
        logger.info("[PhysxOVVisualizer] Initialized")

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
        if self._sim is None or self._app_iface is None:
            return

        # check if we need to change the render mode
        if mode is not None:
            self.set_render_mode(mode)
        # note: we don't call super().render() anymore because they do above operation inside
        #  and we don't want to do it twice. We may remove it once we drop support for Isaac Sim 2022.2.
        self._sim.set_setting("/app/player/playSimulations", False)
        self._app_iface.update()
        self._sim.set_setting("/app/player/playSimulations", True)

        # app.update() may be changing the cuda device, so we force it back to our desired device here
        if "cuda" in self._sim.device:
            torch.cuda.set_device(self._sim.device)

    @property
    def has_gui(self) -> bool:
        """Whether any GUI is available (local, livestreamed, or XR)."""
        if self._sim is None:
            return False
        return bool(self._sim.carb_settings.get("/isaaclab/has_gui"))

    @property
    def app(self) -> omni.kit.app.IApp:
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

    # ------------------------------------------------------------------
    # Timeline Control
    # ------------------------------------------------------------------

    @property
    def timeline(self) -> TimelineControl | None:
        """Get the timeline control instance."""
        return self._timeline

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

    # ------------------------------------------------------------------
    # Visualizer Interface Implementation
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Check if visualizer is still running (app is alive, not whether playing)."""
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
            logger.warning("[PhysxOVVisualizer] Cannot set camera view - visualizer not initialized.")
            return

        try:
            # Import Isaac Sim viewport utilities
            import isaacsim.core.utils.viewports as vp_utils

            # Get the camera prim path - use viewport context or config default
            camera_path = self.cfg.camera_prim_path

            # Use Isaac Sim utility to set camera view
            vp_utils.set_camera_view(
                eye=list(eye), target=list(target), camera_prim_path=camera_path
            )

            logger.info(f"[PhysxOVVisualizer] Camera set: pos={eye}, target={target}, camera={camera_path}")

        except Exception as e:
            logger.warning(f"[PhysxOVVisualizer] Could not set camera: {e}")

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
        if self._sim is None:
            return

        # check if mode change is possible -- not possible when no GUI is available
        if not self._sim.carb_settings.get("/isaaclab/has_gui"):
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

    def reset(self, soft: bool = False) -> None:
        """Reset visualizer (timeline control + warmup renders on hard reset).

        Args:
            soft: If True, skip timeline reset and warmup.
        """
        if self._timeline is None:
            return

        if not soft:
            # disable app control on stop handle during reset
            self._timeline.set_stop_handle_enabled(False)
            if not self.is_stopped():
                self.stop()
            self._timeline.set_stop_handle_enabled(True)
            # play the simulation
            self.play()
            # warmup renders to initialize replicator buffers
            warmup_count = self.cfg.warmup_renders if hasattr(self.cfg, "warmup_renders") else 2
            for _ in range(warmup_count):
                self.render()

    def close(self) -> None:
        """Clean up visualizer resources."""
        # Clean up timeline control
        if self._timeline is not None:
            self._timeline.close()
            self._timeline = None

        self._sim = None
        self._app_iface = None
        self._viewport_context = None
        self._viewport_window = None
        self._is_initialized = False
        self._is_closed = True

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _on_timeline_stop(self) -> None:
        """Callback invoked when the simulation timeline is stopped.

        Once the simulation is stopped, the physics handles go invalid. After that, it is not possible to
        resume the simulation from the last state. This leaves the app in an inconsistent state, where
        two possible actions can be taken:

        1. **Keep the app rendering**: In this case, the simulation is kept running and the app is not shutdown.
           However, the physics is not updated and the script cannot be resumed from the last state. The
           user has to manually close the app to stop the simulation.
        2. **Shutdown the app**: This is the default behavior. In this case, the app is shutdown and
           the simulation is stopped.

        Note:
            This callback is used only when running the simulation in a standalone python script. In an extension,
            it is expected that the user handles the extension shutdown.
        """
        # Currently a no-op, but can be extended to handle stop events
        pass

    def _apply_render_settings_from_cfg(self):  # noqa: C901
        """Sets rtx settings specified in the RenderCfg."""
        if self._sim is None:
            return

        # define mapping of user-friendly RenderCfg names to native carb names
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
        }

        not_carb_settings = ["rendering_mode", "carb_settings", "antialiasing_mode"]

        # grab the rendering mode using the following priority:
        # 1. command line argument --rendering_mode, if provided
        # 2. rendering_mode from Render Config, if set
        # 3. lastly, default to "balanced" mode, if neither is specified
        rendering_mode = self._sim.carb_settings.get("/isaaclab/rendering/rendering_mode")
        if not rendering_mode:
            rendering_mode = self._sim.cfg.render.rendering_mode
        if not rendering_mode:
            rendering_mode = "balanced"

        # set preset settings (same behavior as the CLI arg --rendering_mode)
        if rendering_mode is not None:
            # check if preset is supported
            supported_rendering_modes = ["performance", "balanced", "quality"]
            if rendering_mode not in supported_rendering_modes:
                raise ValueError(
                    f"RenderCfg rendering mode '{rendering_mode}' not in supported modes {supported_rendering_modes}."
                )

            # grab isaac lab apps path
            isaaclab_app_exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "apps")
            # for Isaac Sim 4.5 compatibility, we use the 4.5 rendering mode app files in a different folder
            if get_isaac_sim_version().major < 5:
                isaaclab_app_exp_path = os.path.join(isaaclab_app_exp_path, "isaacsim_4_5")

            # grab preset settings
            preset_filename = os.path.join(isaaclab_app_exp_path, f"rendering_modes/{rendering_mode}.kit")
            with open(preset_filename) as file:
                preset_dict = toml.load(file)
            preset_dict = dict(flatdict.FlatDict(preset_dict, delimiter="."))

            # set presets
            for key, value in preset_dict.items():
                key = "/" + key.replace(".", "/")  # convert to carb setting format
                self._sim.set_setting(key, value)

        # set user-friendly named settings
        for key, value in vars(self._sim.cfg.render).items():
            if value is None or key in not_carb_settings:
                # skip unset settings and non-carb settings
                continue
            if key not in rendering_setting_name_mapping:
                raise ValueError(
                    f"'{key}' in RenderCfg not found. Note: internal 'rendering_setting_name_mapping' dictionary might"
                    " need to be updated."
                )
            key = rendering_setting_name_mapping[key]
            self._sim.set_setting(key, value)

        # set general carb settings
        carb_settings = self._sim.cfg.render.carb_settings
        if carb_settings is not None:
            for key, value in carb_settings.items():
                if "_" in key:
                    key = "/" + key.replace("_", "/")  # convert from python variable style string
                elif "." in key:
                    key = "/" + key.replace(".", "/")  # convert from .kit file style string
                if self._sim.get_setting(key) is None:
                    raise ValueError(f"'{key}' in RenderCfg.general_parameters does not map to a carb setting.")
                self._sim.set_setting(key, value)

        # set denoiser mode
        if self._sim.cfg.render.antialiasing_mode is not None:
            try:
                import omni.replicator.core as rep

                rep.settings.set_render_rtx_realtime(antialiasing=self._sim.cfg.render.antialiasing_mode)
            except Exception:
                pass

        # WAR: Ensure /rtx/renderMode RaytracedLighting is correctly cased.
        if self._sim.carb_settings.get("/rtx/rendermode").lower() == "raytracedlighting":
            self._sim.carb_settings.set_string("/rtx/rendermode", "RaytracedLighting")

    def _configure_rendering_dt(self):
        """Configures the rendering/timeline rate based on physics dt and render interval."""
        if self._sim is None or self._timeline is None:
            return

        from pxr import Usd

        cfg = self._sim.cfg
        stage = self._sim.stage
        carb_settings = self._sim.carb_settings

        # compute rendering frequency
        render_interval = max(cfg.render_interval, 1)
        rendering_hz = int(1.0 / (cfg.physics_manager_cfg.dt * render_interval))

        # If rate limiting is enabled, set the rendering rate to the specified value
        # Otherwise run the app as fast as possible and do not specify the target rate
        if carb_settings.get("/app/runLoops/main/rateLimitEnabled"):
            carb_settings.set_int("/app/runLoops/main/rateLimitFrequency", rendering_hz)
            self._timeline.set_target_framerate(rendering_hz)

        # set stage time codes per second
        with Usd.EditContext(stage, stage.GetRootLayer()):
            stage.SetTimeCodesPerSecond(rendering_hz)
        self._timeline.set_time_codes_per_second(rendering_hz)

        # The isaac sim loop runner is enabled by default in isaac sim apps,
        # but in case we are in an app with the kit loop runner, protect against this
        try:
            import omni.kit.loop._loop as omni_loop

            _loop_runner = omni_loop.acquire_loop_interface()
            _loop_runner.set_manual_step_size(cfg.physics_manager_cfg.dt * render_interval)
            _loop_runner.set_manual_mode(True)
        except Exception:
            logger.warning(
                "Isaac Sim loop runner not found, enabling rate limiting to support rendering at specified rate"
            )
            carb_settings.set_bool("/app/runLoops/main/rateLimitEnabled", True)
            carb_settings.set_int("/app/runLoops/main/rateLimitFrequency", rendering_hz)
            self._timeline.set_target_framerate(rendering_hz)
