# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer interface for SimulationContext."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from isaaclab.visualizers import Visualizer

from .interface import Interface
from isaaclab.visualizers.physx_ov_visualizer_cfg import PhysxOVVisualizerCfg

if TYPE_CHECKING:
    from .simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class VisualizerInterface(Interface):
    """Manages visualizer lifecycle and rendering for SimulationContext."""

    def __init__(self, sim_context: "SimulationContext"):
        """Initialize visualizer interface.

        Args:
            sim_context: Parent simulation context.
        """
        super().__init__(sim_context)
        self.dt = self._sim.cfg.physics_manager_cfg.dt * self._sim.cfg.render_interval

        # Visualizer state
        visualizers = "omniverse"
        self._visualizers_str = [v.strip() for v in visualizers.split(",") if v.strip()]
        self._visualizers: list[Visualizer] = []
        self._visualizer_step_counter = 0
        self._scene_data_provider: SceneDataProvider | None = None
        self._was_playing = False

        # Initialize visualizers immediately
        self.initialize_visualizers()

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
    def visualizers(self) -> list[Visualizer]:
        return self._visualizers

    @property
    def scene_data_provider(self) -> SceneDataProvider | None:
        return self._scene_data_provider

    # -- Visualizer Initialization --

    def _create_default_visualizer_configs(self, requested: list[str]) -> list:
        """Create default configs for requested visualizer types."""
        configs = []
        type_map = {"omniverse": PhysxOVVisualizerCfg}

        for viz_type in requested:
            if viz_type in type_map:
                try:
                    configs.append(type_map[viz_type]())
                except Exception as e:
                    logger.error(f"Failed to create default config for '{viz_type}': {e}")
            else:
                logger.warning(f"Unknown visualizer type '{viz_type}'. Valid: {list(type_map.keys())}")

        return configs

    def initialize_visualizers(self) -> None:
        """Initialize visualizers based on --visualizer flag."""
        if not self._visualizers_str:
            if bool(self.settings.get("/isaaclab/visualizer")) or bool(self.settings.get("/isaaclab/render/offscreen")):
                logger.info("No visualizers specified via --visualizer flag.")
            return

        # Get or create visualizer configs
        cfg_list = self._sim.cfg.visualizer_cfgs
        if cfg_list is None:
            visualizer_cfgs = self._create_default_visualizer_configs(self._visualizers_str)
        else:
            visualizer_cfgs = cfg_list if isinstance(cfg_list, list) else [cfg_list]
            visualizer_cfgs = [c for c in visualizer_cfgs if c.visualizer_type in self._visualizers_str]

            if not visualizer_cfgs:
                logger.info(f"Creating default configs for: {self._visualizers_str}")
                visualizer_cfgs = self._create_default_visualizer_configs(self._visualizers_str)

        if not visualizer_cfgs:
            return

        # Create scene data provider
        self._scene_data_provider = None  # SceneDataProvider(visualizer_cfgs)

        # Initialize each visualizer
        for cfg in visualizer_cfgs:
            try:
                visualizer = cfg.create_visualizer()
                scene_data = self._build_scene_data(cfg)
                visualizer.initialize(scene_data)
                self._visualizers.append(visualizer)
                logger.info(f"Initialized: {type(visualizer).__name__} ({cfg.visualizer_type})")
            except Exception as e:
                logger.error(f"Failed to init '{cfg.visualizer_type}': {e}")

    def _build_scene_data(self, cfg) -> dict:
        """Build scene data dict for visualizer initialization."""
        if cfg.visualizer_type in ("newton", "rerun"):
            return {"scene_data_provider": self._scene_data_provider}
        elif cfg.visualizer_type == "omniverse":
            return {"usd_stage": self._sim.stage, "simulation_context": self._sim}
        return {}

    # -- Unified Interface Methods --

    def is_playing(self) -> bool:
        """Check whether the simulation is playing."""
        for viz in self.visualizers:
            return viz.is_running()
        return True  # physics is always playing when there is no visualizer, basically headless mode

    def is_stopped(self) -> bool:
        """Check whether the simulation is stopped (not paused)."""
        for viz in self.visualizers:
            return viz.is_stopped()
        return False

    def forward(self) -> None:
        """Sync scene data and step all active visualizers.

        Args:
            dt: Time step in seconds (0.0 for kinematics-only).
        """
        if self._scene_data_provider:
            self._scene_data_provider.update()

        if not self._visualizers:
            return

    def step(self, render: bool = True) -> None:
        """Step visualizers and optionally render.

        Args:
            render: Whether to render after stepping.
        """
        # If paused, keep rendering until playing or stopped
        while not self.is_playing() and not self.is_stopped():
            self.render()
            self._was_playing = False

        # Detect transition: was not playing → now playing (resume from pause)
        is_playing = self.is_playing()
        if not self._was_playing and is_playing:
            self.reset(soft=True)

        # Update state tracking
        self._was_playing = is_playing

        self.forward()

        if render:
            self.render()

    def reset(self, soft: bool) -> None:
        """Reset visualizers (warmup renders on hard reset)."""
        for viz in self._visualizers:
            viz.reset(soft)

    def close(self) -> None:
        """Close all visualizers and clean up."""
        for viz in self._visualizers:
            try:
                viz.close()
            except Exception as e:
                logger.error(f"Error closing {type(viz).__name__}: {e}")

        self._visualizers.clear()
        logger.info("All visualizers closed")

    def play(self) -> None:
        """Handle simulation start."""
        for viz in self._visualizers:
            viz.play()

    def stop(self) -> None:
        """Handle simulation stop."""
        for viz in self._visualizers:
            viz.stop()

    def pause(self) -> None:
        """Pause the simulation."""
        for viz in self._visualizers:
            viz.pause()

    def render(self) -> None:
        """Render the scene.

        Args:
            mode: Render mode to set, or None to keep current.
        """
        self._visualizer_step_counter += 1
        to_remove = []

        for viz in self._visualizers:
            try:
                if not viz.is_running():
                    to_remove.append(viz)
                    continue

                # Block while training paused
                while viz.is_training_paused() and viz.is_running():
                    viz.step(0.0, state=None)

                viz.step(self.get_rendering_dt() or self.dt, state=None)
            except Exception as e:
                logger.error(f"Error stepping {type(viz).__name__}: {e}")
                to_remove.append(viz)

        for viz in to_remove:
            try:
                viz.close()
                self._visualizers.remove(viz)
                logger.info(f"Removed: {type(viz).__name__}")
            except Exception as e:
                logger.error(f"Error closing visualizer: {e}")

    def get_rendering_dt(self) -> float:
        """Get rendering dt from visualizers, or fall back to physics dt."""
        for viz in self._visualizers:
            dt = viz.get_rendering_dt()
            if dt is not None:
                return dt
        return self.dt

    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        """Set camera view on all visualizers that support it."""
        for viz in self._visualizers:
            viz.set_camera_view(eye, target)
