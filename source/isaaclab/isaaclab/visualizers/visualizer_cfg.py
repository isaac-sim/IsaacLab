# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for visualizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .visualizer import Visualizer


@configclass
class VisualizerCfg:
    """Base configuration for all visualizer backends.

    Note:
        This is an abstract base class and should not be instantiated directly.
        Use specific visualizer configs like NewtonVisualizerCfg, RerunVisualizerCfg, or KitVisualizerCfg.
    """

    visualizer_type: str | None = None
    """Type identifier (e.g., 'newton', 'rerun', 'kit'). Must be overridden by subclasses."""

    enable_markers: bool = True
    """Enable visualization markers (debug drawing)."""

    enable_live_plots: bool = True
    """Enable live plotting of data."""

    camera_position: tuple[float, float, float] = (8.0, 8.0, 3.0)
    """Initial camera position (x, y, z) in world coordinates."""

    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target/look-at point (x, y, z) in world coordinates."""

    camera_source: Literal["cfg", "usd_path"] = "cfg"
    """Camera source mode: 'cfg' uses camera_position/target, 'usd_path' follows a USD camera prim."""

    camera_usd_path: str = "/World/envs/env_0/Camera"
    """Absolute USD path to a camera prim when camera_source='usd_path'."""

    env_filter_mode: Literal["none", "env_ids", "random_n"] = "none"
    """Env filter mode: 'none', 'env_ids', or 'random_n'."""

    env_filter_random_n: int = 64
    """If env_filter_mode='random_n', number of envs to sample."""

    env_filter_seed: int = 0
    """Seed for deterministic env sampling."""

    env_filter_ids: list[int] = [i for i in range(0, 64, 4)]
    """If env_filter_mode='env_ids', only these env indices are shown.

    This improves performance, particularly for large-scale training, by reducing scene updates sent to visualizers.
    Note, OV visualizer only applies a cosmetic visibility toggle (no performance gain).
    """

    def get_visualizer_type(self) -> str | None:
        """Get the visualizer type identifier.

        Returns:
            The visualizer type string, or None if not set (base class).
        """
        return self.visualizer_type

    def create_visualizer(self) -> Visualizer:
        """Create visualizer instance from this config using factory pattern.

        Raises:
            ValueError: If visualizer_type is None (base class used directly) or not registered.
        """
        from . import get_visualizer_class

        if self.visualizer_type is None:
            raise ValueError(
                "Cannot create visualizer from base VisualizerCfg class. "
                "Use a specific visualizer config: NewtonVisualizerCfg, RerunVisualizerCfg, or KitVisualizerCfg."
            )

        visualizer_class = get_visualizer_class(self.visualizer_type)
        if visualizer_class is None:
            if self.visualizer_type in ("newton", "rerun"):
                raise ImportError(
                    f"Visualizer '{self.visualizer_type}' requires the Newton Python module and its dependencies. "
                    "Install the Newton backend (e.g., newton package/isaaclab_newton) and retry."
                )
            raise ValueError(
                f"Visualizer type '{self.visualizer_type}' is not registered. Valid types: 'newton', 'rerun', 'kit'."
            )

        return visualizer_class(self)
