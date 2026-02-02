# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer interface for SimulationContext."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .interface import Interface
from .ov_visualizer import OVVisualizer, RenderMode

if TYPE_CHECKING:
    from .simulation_context import SimulationContext

# Re-export RenderMode for backwards compatibility
__all__ = ["RenderMode", "VisualizerInterface"]


class VisualizerInterface(Interface):
    """Manages visualizer lifecycle and rendering for SimulationContext."""

    # Expose RenderMode as class attribute for backwards compatibility
    RenderMode = RenderMode

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize visualizer interface.

        Args:
            sim_context: Parent simulation context.
        """
        super().__init__(sim_context)
        self.dt = self._sim.cfg.dt * self._sim.cfg.render_interval
        # Create OV visualizer helper
        self._ov_visualizer = OVVisualizer(sim_context)
        # Track previous playing state to detect transitions
        self._was_playing = False

    # -- Properties --

    @property
    def settings(self):
        return self._sim.carb_settings

    @property
    def device(self) -> str:
        return self._sim.device

    @property
    def stage(self):
        return self._sim.stage

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
    # ------------------------------------------------------------------
    # Timeline Control (delegate to OVVisualizer)
    # ------------------------------------------------------------------

    def is_playing(self) -> bool:
        """Check whether the simulation is playing."""
        return self._ov_visualizer.is_playing()

    def is_stopped(self) -> bool:
        """Check whether the simulation is stopped."""
        return self._ov_visualizer.is_stopped()

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


    def forward(self) -> None:
        """Sync scene data and step all active visualizers.

        Args:
            dt: Time step in seconds (0.0 for kinematics-only).
        """
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

    def reset(self, soft: bool) -> None:
        """Reset visualizers (warmup renders on hard reset)."""
        self._ov_visualizer.reset(soft)

    def close(self) -> None:
        """Close all visualizers and clean up."""
        self._ov_visualizer.close()

    def play(self) -> None:
        """Handle simulation start."""
        self._ov_visualizer.play()

    def stop(self) -> None:
        """Handle simulation stop."""
        self._ov_visualizer.stop()

    def pause(self) -> None:
        """Pause the simulation."""
        self._ov_visualizer.pause()

    def render(self, mode: RenderMode | None = None):
        """Render the scene (OV mode only).

        Args:
            mode: Render mode to set, or None to keep current.

        Returns:
            True if rendered, False if not in OV mode.
        """
        self._ov_visualizer.render(mode)


    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        """Set camera view on all visualizers that support it."""
        self._ov_visualizer.set_camera_view(eye, target, "/OmniverseKit_Persp")
