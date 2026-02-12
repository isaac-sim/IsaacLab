# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for visualizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .visualizer_cfg import VisualizerCfg


class Visualizer(ABC):
    """Base class for all visualizer backends.

    Lifecycle: __init__() -> initialize() -> step() (repeated) -> close()
    """

    def __init__(self, cfg: VisualizerCfg):
        """Initialize visualizer with config."""
        self.cfg = cfg
        self._is_initialized = False
        self._is_closed = False

    @abstractmethod
    def initialize(self, scene_data: dict[str, Any] | None = None) -> None:
        """Initialize visualizer with scene data (model, state, usd_stage, etc.)."""
        pass

    @abstractmethod
    def step(self, dt: float, state: Any | None = None) -> None:
        """Update visualization for one step.

        Args:
            dt: Time step in seconds.
            state: Updated physics state (e.g., newton.State).
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if visualizer is still running (e.g., window not closed)."""
        pass

    @abstractmethod
    def is_stopped(self) -> bool:
        """Check if visualizer is stopped (e.g., window closed)."""
        pass

    def is_training_paused(self) -> bool:
        """Check if training is paused by visualizer controls."""
        return False

    def is_rendering_paused(self) -> bool:
        """Check if rendering is paused by visualizer controls."""
        return False

    @property
    def is_initialized(self) -> bool:
        """Check if initialize() has been called."""
        return self._is_initialized

    @property
    def is_closed(self) -> bool:
        """Check if close() has been called."""
        return self._is_closed

    def supports_markers(self) -> bool:
        """Check if visualizer supports VisualizationMarkers."""
        return False

    def supports_live_plots(self) -> bool:
        """Check if visualizer supports LivePlots."""
        return False

    def get_rendering_dt(self) -> float | None:
        """Get rendering time step. Returns None to use interface default."""
        return None

    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        """Set camera view position. No-op by default."""
        pass

    def reset(self, soft: bool = False) -> None:
        """Reset visualizer state. No-op by default."""
        pass

    def play(self) -> None:
        """Handle simulation play/start. No-op by default."""
        pass

    def pause(self) -> None:
        """Handle simulation pause. No-op by default."""
        pass

    def stop(self) -> None:
        """Handle simulation stop. No-op by default."""
        pass
