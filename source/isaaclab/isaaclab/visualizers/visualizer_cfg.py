# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for visualizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .base_visualizer import BaseVisualizer


@configclass
class VisualizerCfg:
    """Base configuration for all visualizer backends.

    Note:
        This is an abstract base class and should not be instantiated directly.
        Use specific configs from isaaclab_visualizers: KitVisualizerCfg, NewtonVisualizerCfg,
        RerunVisualizerCfg, or ViserVisualizerCfg (from isaaclab_visualizers.kit/.newton/.rerun/.viser).
    """

    visualizer_type: str | None = None
    """Type identifier (e.g., 'newton', 'rerun', 'viser', 'kit'). Must be overridden by subclasses."""

    requires_newton_model: bool = False
    """Internal requirement flag for scene-data setup; avoid overriding in user configs."""

    requires_usd_stage: bool = False
    """Internal requirement flag for scene-data setup; avoid overriding in user configs."""

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

    def create_visualizer(self) -> BaseVisualizer:
        """Create visualizer instance from this config using factory pattern.

        Loads the matching backend from isaaclab_visualizers (e.g. isaaclab_visualizers.rerun).

        Raises:
            ValueError: If visualizer_type is None (base class used directly) or not registered.
            ImportError: If isaaclab_visualizers or the requested backend extra is not installed.
        """
        from .visualizer import Visualizer

        if self.visualizer_type is None:
            raise ValueError(
                "Cannot create visualizer from base VisualizerCfg class. "
                "Use a specific config from isaaclab_visualizers "
                "(e.g. KitVisualizerCfg, NewtonVisualizerCfg, RerunVisualizerCfg, ViserVisualizerCfg)."
            )

        try:
            return Visualizer(self)
        except (ValueError, ImportError, ModuleNotFoundError) as exc:
            if self.visualizer_type in ("newton", "rerun", "viser", "kit"):
                raise ImportError(
                    f"Visualizer '{self.visualizer_type}' requires the isaaclab_visualizers package. "
                    f"Install with: pip install isaaclab_visualizers[{self.visualizer_type}]"
                ) from exc
            raise
