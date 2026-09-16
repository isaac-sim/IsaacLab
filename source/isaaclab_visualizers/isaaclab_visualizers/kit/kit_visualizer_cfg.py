# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Kit-based visualizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.utils import config_field
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

if TYPE_CHECKING:
    from .kit_visualizer import KitVisualizer


@dataclass
class KitVisualizerCfg(VisualizerCfg):
    """Configuration for Kit visualizer using Isaac Sim viewport.

    .. note::
        The streaming camera panel (``streaming_view=True``) requires the
        ``--enable_cameras`` CLI flag.  Without it, the streaming view is silently
        skipped and no image panel is created.  Set ``dock_position="RIGHT"`` so
        the panel appears side-by-side with the Viewport instead of as a hidden tab.
    """

    class_type: type[KitVisualizer] | str = config_field("{DIR}.kit_visualizer:KitVisualizer")
    """Visualizer implementation class."""

    visualizer_type: str = config_field("kit")
    """Type identifier for Kit visualizer."""

    viewport_name: str | None = config_field(None)
    """Name for a new viewport window when :attr:`create_viewport` is ``True``.

    If ``None``, a default name (``"Visualizer Viewport"``) is used.
    """

    create_viewport: bool = config_field(False)
    """If ``True``, create a new viewport window; if ``False``, use the active viewport window."""

    headless: bool = config_field(False)
    """Run without creating viewport windows when supported by the app."""

    dock_position: str = config_field("SAME")
    """Dock position for the streaming image panel and any new viewport window.

    Options: ``'LEFT'``, ``'RIGHT'``, ``'BOTTOM'``, ``'SAME'``.

    .. note::
        ``'SAME'`` (the default) places the streaming panel as a hidden tab in the
        same dock group as the main Viewport — you must click the panel's tab to see it.
        Use ``'RIGHT'`` to keep both the Viewport and the streaming panel visible
        side-by-side.
    """

    window_width: int = config_field(1280)
    """Viewport width in pixels (when :attr:`create_viewport` is ``True``)."""

    window_height: int = config_field(720)
    """Viewport height in pixels (when :attr:`create_viewport` is ``True``)."""

    origin_type: str = config_field("world")
    """Frame in which :attr:`~isaaclab.visualizers.VisualizerCfg.eye` and
    :attr:`~isaaclab.visualizers.VisualizerCfg.lookat` are interpreted.

    Options:

    * ``"world"``: global origin.
    * ``"env"``: origin of the environment at :attr:`origin_env_index`.
    * ``"asset"``: a scene asset (or body) specified by :attr:`origin_track_path`.
    """

    origin_env_index: int = config_field(0)
    """Index of the environment used as the viewport camera origin.

    Only meaningful when :attr:`origin_type` is ``"env"`` or ``"asset"``.
    """

    origin_track_path: str | None = config_field(None)
    """Asset tracking path for the viewport camera origin.

    Format: ``"<asset_name>"`` to track the asset root, or ``"<asset_name>/<body_name>"``
    to track a specific body on the asset.  Required when :attr:`origin_type` is ``"asset"``.

    Examples::

        origin_track_path = "robot"             # track robot root
        origin_track_path = "robot/panda_hand"  # track panda_hand body on robot
    """
