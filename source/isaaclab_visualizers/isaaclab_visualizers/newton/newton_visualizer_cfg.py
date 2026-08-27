# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for Newton GL and RTX visualizer backends."""

from __future__ import annotations

import warnings

from isaaclab.utils.configclass import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg


@configclass
class NewtonVisualizerCfg(VisualizerCfg):
    """Shared configuration base for Newton visualizer backends.

    .. deprecated::
        :class:`NewtonVisualizerCfg` is deprecated. Use :class:`NewtonGLVisualizerCfg` for the
        OpenGL rasterizer or :class:`NewtonRTXVisualizerCfg` for the OVRTX path tracer.
    """

    # Deprecated alias: "newton" routes to the GL backend via simulation_context._VISUALIZER_ALIASES.
    visualizer_type: str = "newton_gl"

    def __post_init__(self):
        if type(self) is NewtonVisualizerCfg:
            warnings.warn(
                "NewtonVisualizerCfg is deprecated and will be removed in a future release. "
                "Use NewtonGLVisualizerCfg (OpenGL rasterizer) or NewtonRTXVisualizerCfg (OVRTX path tracer) instead.",
                DeprecationWarning,
                stacklevel=3,
            )

    window_width: int = 1920
    """Window width in pixels."""

    window_height: int = 1080
    """Window height in pixels."""

    headless: bool = False
    """Run the Newton viewer without requiring a display server."""

    update_frequency: int = 1
    """Visualizer update frequency (renders every N simulation frames)."""

    world_spacing: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Visual spacing between simulation worlds along each axis [m].

    Non-zero axes arrange visible worlds in a compact grid without changing their simulated poses.
    """

    show_joints: bool = False
    """Show joint visualization."""

    show_contacts: bool = False
    """Show contact visualization."""

    show_collision: bool = False
    """Show collision visualization."""

    show_springs: bool = False
    """Show spring visualization."""

    show_inertia_boxes: bool = False
    """Show inertia box visualization."""

    show_com: bool = False
    """Show center of mass visualization."""

    show_particles: bool = False
    """Show particle visualization."""

    particle_color: tuple[float, float, float] | None = None
    """Optional particle color RGB [0, 1]. Uses Newton viewer defaults when ``None``."""

    enable_picking: bool = True
    """Enable right-click dragging with Newton rigid-body solvers.

    Supported coupled solvers may expose dragging through a rigid-body entry.
    Disabled automatically for headless viewers, standalone MPM, and non-Newton
    physics. MPM particles are not pickable.
    """

    enable_shadows: bool = True
    """Enable shadow rendering."""

    enable_sky: bool = True
    """Enable sky rendering."""

    enable_wireframe: bool = False
    """Enable wireframe rendering."""

    sky_upper_color: tuple[float, float, float] = (0.2, 0.4, 0.6)
    """Sky upper color RGB [0, 1]."""

    sky_lower_color: tuple[float, float, float] = (0.5, 0.6, 0.7)
    """Sky lower color RGB [0, 1]."""

    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Light color RGB [0, 1]."""


@configclass
class NewtonGLVisualizerCfg(NewtonVisualizerCfg):
    """Configuration for the Newton OpenGL rasterizer visualizer.

    Selects Newton's OpenGL backend — fast local window with the full Isaac Lab
    feature set: streaming camera panel, particle color override, and live scalar/array plots.

    The streaming camera panel is enabled by default (``streaming_view=True``) but starts
    hidden — no camera rendering work is performed until the user opens the panel via the
    sidebar combo, keeping per-step overhead zero when the panel is closed.
    """

    visualizer_type: str = "newton_gl"
    """Factory type identifier. Do not change."""

    streaming_view: bool = True
    """Enable the tiled streaming camera panel.

    Overrides the base-class default of ``False``.  The panel starts **hidden** so there
    is no per-step camera rendering cost; the user can open it at any time via the
    *Streaming View* combo in the Newton sidebar.
    """


@configclass
class NewtonRTXVisualizerCfg(NewtonVisualizerCfg):
    """Configuration for the Newton OVRTX path-tracer visualizer.

    Selects Newton's OVRTX backend — photorealistic rendering using the same
    ``begin_frame / log_state / end_frame`` step interface as the GL backend.

    .. note::
        RTX render quality settings (fps, lighting environment, denoiser, etc.)
        are not yet exposed here; ``ViewerRTX`` defaults are used. These will be
        surfaced in a future revision in a way that is consistent across all
        RTX-capable renderers.

    ``render_rgb_array()`` captures the path-traced LDR framebuffer at
    :attr:`window_width` by :attr:`window_height`. The tiled camera panel remains
    unsupported because ``ViewerRTX.log_image`` has no display sink.
    """

    visualizer_type: str = "newton_rtx"
    """Factory type identifier. Do not change."""

    rtx_environment: str = "default"
    """OVRTX lighting environment.  One of ``"default"`` (dome + distant light),
    ``"studio"`` (three-point rig for cleaner highlights), or ``"none"``."""
