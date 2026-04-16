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

    enable_markers: bool = True
    """Enable visualization markers (debug drawing)."""

    enable_live_plots: bool = True
    """Enable live plotting of data."""

    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Initial camera eye position (x, y, z) in world coordinates."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look-at point (x, y, z) in world coordinates."""

    cam_source: Literal["cfg", "prim_path"] = "cfg"
    """Camera source mode: 'cfg' uses eye/lookat, 'prim_path' follows a camera prim."""

    cam_prim_path: str = "/World/envs/env_0/Camera"
    """Absolute USD path to a camera prim when cam_source='prim_path'."""

    env_selection_max_visible: int | None = 4
    """When ``env_selection_mode`` is ``none``, optional cap on how many envs are shown (``0..num_envs-1``)."""

    env_selection_mode: Literal["none", "env_ids", "random_n"] = "none"
    """How env indices are chosen for viewers: ``none`` (use :attr:`env_selection_max_visible` only), ``env_ids``, or ``random_n``."""

    env_selection_ids: list[int] = [i for i in range(0, 64, 4)]
    """When ``env_selection_mode`` is ``env_ids``, only these env indices are shown.
    """

    env_selection_random_count: int = 64
    """When ``env_selection_mode`` is ``random_n``, number of env indices to sample."""

    env_selection_random_seed: int = 0
    """Seed for deterministic sampling when ``env_selection_mode`` is ``random_n``."""

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
