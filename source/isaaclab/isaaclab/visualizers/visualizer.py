# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for visualizers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.sim.scene_data_providers import SceneDataProvider

    from .visualizer_cfg import VisualizerCfg


class Visualizer(ABC):
    """Base class for all visualizer backends.

    Lifecycle: __init__() -> initialize() -> step() (repeated) -> close()
    """

    def __init__(self, cfg: VisualizerCfg):
        """Initialize visualizer with config."""
        self.cfg = cfg
        self._scene_data_provider = None
        self._is_initialized = False
        self._is_closed = False
        self._env_ids: list[int] | None = None  # env indices to show; None = all

    @abstractmethod
    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Initialize visualizer resources."""
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float, state: Any | None = None) -> None:
        """Update visualization for one step.

        Args:
            dt: Time step in seconds.
            state: Updated physics state (e.g., newton.State).
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        raise NotImplementedError

    @abstractmethod
    def is_running(self) -> bool:
        """Check if visualizer is still running (e.g., window not closed)."""
        raise NotImplementedError

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

    def get_visualized_env_ids(self) -> list[int] | None:
        """Return env indices this visualizer is showing. None = all envs (no partial viz)."""
        return self._env_ids

    def _compute_visualized_env_ids(self) -> list[int] | None:
        """Compute which env indices to show from config.

        If env_ids is set, only those envs are shown. Otherwise, show all envs.
        """
        if self._scene_data_provider is None:
            return None
        cfg = self.cfg
        num_envs = self._scene_data_provider.get_metadata().get("num_envs", 0)
        if num_envs <= 0:
            logger.warning(
                "[Visualizer] num_envs is 0 or missing from provider metadata; partial visualization disabled."
            )
            return None
        env_ids_cfg = getattr(cfg, "env_ids", None)
        if env_ids_cfg is not None and len(env_ids_cfg) > 0:
            return [i for i in env_ids_cfg if 0 <= i < num_envs]
        return None
