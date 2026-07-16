# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Kit-based visualizer."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg


@configclass
class KitVisualizerCfg(VisualizerCfg):
    """Configuration for Kit visualizer using Isaac Sim viewport."""

    visualizer_type: str = "kit"
    """Type identifier for Kit visualizer."""

    viewport_name: str | None = None
    """Name for a new viewport window when :attr:`create_viewport` is ``True``.

    If ``None``, a default name (``"Visualizer Viewport"``) is used.
    """

    create_viewport: bool = False
    """If ``True``, create a new viewport window; if ``False``, use the active viewport window."""

    headless: bool = False
    """Run without creating viewport windows when supported by the app."""

    dock_position: str = "SAME"
    """Dock position for a new viewport. Options: 'LEFT', 'RIGHT', 'BOTTOM', 'SAME'."""

    window_width: int = 1280
    """Viewport width in pixels (when :attr:`create_viewport` is ``True``)."""

    window_height: int = 720
    """Viewport height in pixels (when :attr:`create_viewport` is ``True``)."""

    origin_type: str = "world"
    """Frame in which :attr:`~isaaclab.visualizers.VisualizerCfg.eye` and
    :attr:`~isaaclab.visualizers.VisualizerCfg.lookat` are interpreted.

    Options:

    * ``"world"``: global origin.
    * ``"env"``: origin of the environment at :attr:`env_index`.
    * ``"asset_root"``: root of the asset named :attr:`asset_name` in environment :attr:`env_index`.
    * ``"asset_body"``: specific body of the asset (requires :attr:`asset_name` and :attr:`body_name`).
    """

    env_index: int = 0
    """Index of the environment used as the camera origin.

    Only meaningful when :attr:`origin_type` is ``"env"``, ``"asset_root"``, or ``"asset_body"``.
    """

    asset_name: str | None = None
    """Name of the asset in the interactive scene used as the camera tracking target.

    Required when :attr:`origin_type` is ``"asset_root"`` or ``"asset_body"``.
    """

    body_name: str | None = None
    """Name of the body within :attr:`asset_name` used as the camera tracking target.

    Required when :attr:`origin_type` is ``"asset_body"``.
    """
