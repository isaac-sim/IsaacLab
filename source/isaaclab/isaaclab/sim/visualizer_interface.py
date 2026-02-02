# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer interface for SimulationContext (Omniverse/PhysX workflow)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .interface import Interface
from .ov_visualizer import OVVisualizer, RenderMode

if TYPE_CHECKING:
    from .simulation_context import SimulationContext

# Re-export RenderMode for backwards compatibility
__all__ = ["RenderMode", "VisualizerInterface"]


class VisualizerInterface(Interface):
    """Manages viewport/rendering for SimulationContext (Omniverse workflow).

    This interface delegates rendering operations to an internal OVVisualizer helper,
    while managing the lifecycle integration with SimulationContext.
    """

    # Expose RenderMode as class attribute for backwards compatibility
    RenderMode = RenderMode

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize visualizer interface.

        Args:
            sim_context: Parent simulation context.
        """
        super().__init__(sim_context)

        # Create OV visualizer helper
        self._ov_visualizer = OVVisualizer(sim_context)

        # Track previous playing state to detect transitions
        self._was_playing = False

    # ------------------------------------------------------------------
    # Properties (delegate to OVVisualizer)
    # ------------------------------------------------------------------
    @property
    def app(self):
        """Omniverse Kit Application interface."""
        return self._ov_visualizer.app

    @property
    def offscreen_render(self) -> bool:
        """Whether offscreen rendering is enabled."""
        return self._ov_visualizer.offscreen_render

    @property
    def render_viewport(self) -> bool:
        """Whether the default viewport should be rendered."""
        return self._ov_visualizer.render_viewport

    @property
    def render_mode(self) -> RenderMode:
        """Current render mode."""
        return self._ov_visualizer.render_mode

    @render_mode.setter
    def render_mode(self, value: RenderMode) -> None:
        """Set render mode."""
        self._ov_visualizer.render_mode = value

    # ------------------------------------------------------------------
    # Timeline Control (delegate to OVVisualizer)
    # ------------------------------------------------------------------

    def is_playing(self) -> bool:
        """Check whether the simulation is playing."""
        return self._ov_visualizer.is_playing()

    def is_stopped(self) -> bool:
        """Check whether the simulation is stopped."""
        return self._ov_visualizer.is_stopped()

    def play(self) -> None:
        """Start playing the simulation."""
        self._ov_visualizer.play()

    def pause(self) -> None:
        """Pause the simulation."""
        self._ov_visualizer.pause()

    def stop(self) -> None:
        """Stop the simulation."""
        self._ov_visualizer.stop()

    # ------------------------------------------------------------------
    # Render Mode (delegate to OVVisualizer)
    # ------------------------------------------------------------------

    def set_render_mode(self, mode: RenderMode):
        """Change the current render mode of the simulation.

        Please see :class:`RenderMode` for more information on the different render modes.

        Args:
            mode: The rendering mode. If different than current rendering mode,
                the mode is changed to the new mode.
        """
        self._ov_visualizer.set_render_mode(mode)

    # ------------------------------------------------------------------
    # Camera (delegate to OVVisualizer)
    # ------------------------------------------------------------------

    def set_camera_view(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        camera_prim_path: str = "/OmniverseKit_Persp",
    ):
        """Set the location and target of the viewport camera in the stage.

        Args:
            eye: The location of the camera eye.
            target: The location of the camera target.
            camera_prim_path: The path to the camera primitive in the stage. Defaults to
                "/OmniverseKit_Persp".
        """
        self._ov_visualizer.set_camera_view(eye, target, camera_prim_path)

    # ------------------------------------------------------------------
    # Interface Methods
    # ------------------------------------------------------------------

    def render(self, mode: RenderMode | None = None):
        """Refreshes the rendering components including UI elements and view-ports depending on the render mode.

        This function is used to refresh the rendering components of the simulation. This includes updating the
        view-ports, UI elements, and other extensions (besides physics simulation) that are running in the
        background. The rendering components are refreshed based on the render mode.

        Please see :class:`RenderMode` for more information on the different render modes.

        Args:
            mode: The rendering mode. Defaults to None, in which case the current rendering mode is used.
        """
        self._ov_visualizer.render(mode)

    def reset(self, soft: bool = False) -> None:
        """Reset visualizer (timeline control + warmup renders on hard reset).

        Args:
            soft: If True, skip timeline reset and warmup.
        """
        self._ov_visualizer.reset(soft)

    def forward(self) -> None:
        """No-op for visualizer (rendering happens in render())."""
        pass

    def step(self, render: bool = True) -> None:
        """Step visualizers and optionally render.

        Args:
            render: Whether to render after stepping.
        """
        # If paused, keep rendering until playing or stopped
        while not self.is_playing() and not self.is_stopped():
            self.render()
            self._was_playing = False

        # # Detect transition: was not playing → now playing (resume from pause)
        is_playing = self.is_playing()
        if not self._was_playing and is_playing:
            self.reset(soft=True) # TODO: it is currently buggy

        # Update state tracking
        self._was_playing = is_playing

        self.forward()

        if render:
            self.render()

    def close(self) -> None:
        """Clean up visualizer resources."""
        self._ov_visualizer.close()
