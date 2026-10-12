# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for Newton GL and RTX visualizer backends."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

if TYPE_CHECKING:
    from .newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer


@configclass
class NewtonVisualizerCfg(VisualizerCfg):
    """Deprecated configuration base for the Newton GL visualizer.

    .. deprecated::
        :class:`NewtonVisualizerCfg` is deprecated. Use :class:`NewtonGLVisualizerCfg` for the
        OpenGL rasterizer or :class:`NewtonRTXVisualizerCfg` for the OVRTX path tracer.
    """

    class_type: type[NewtonGLVisualizer] | str = "{DIR}.newton_visualizer:NewtonGLVisualizer"
    """Deprecated alias for the Newton GL visualizer implementation."""

    # Deprecated alias: "newton" routes to the GL backend via visualizer_cfg.VISUALIZER_ALIASES.
    visualizer_type: str = "newton_gl"

    cloning_contexts: tuple[type | str, ...] = ("isaaclab_newton.cloner:NewtonReplicateContext",)

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self) is NewtonVisualizerCfg:
            warnings.warn(
                "NewtonVisualizerCfg is deprecated and will be removed in a future release. "
                "Use NewtonGLVisualizerCfg (OpenGL rasterizer) or NewtonRTXVisualizerCfg (OVRTX path tracer) instead.",
                DeprecationWarning,
                stacklevel=3,
            )

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
    """Enable procedural sky rendering when ``background_color`` is ``None``."""

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
    feature set: scene-camera display, particle color override, and live scalar/array plots.

    A scene-camera view replaces perspective rendering. With no explicit camera selection,
    the viewer starts in perspective mode and the sidebar can switch to a scene camera.
    """

    class_type: type[NewtonGLVisualizer] | str = "{DIR}.newton_visualizer:NewtonGLVisualizer"
    """Visualizer implementation class."""

    visualizer_type: str = "newton_gl"
    """Visualizer selector identifier. Do not change."""

    streaming_view: bool = True
    """Make scene cameras available in the view selector.

    A SceneCameraCfg selection enables this automatically. Otherwise the viewer starts in perspective.
    """


@configclass
class NewtonRTXVisualizerCfg(VisualizerCfg):
    """Newton ViewerRTX rendering a simulation-owned OVStage.

    Perspective sources use Newton's fixed-resolution render product; scene sources borrow sensor output.
    Window resizing scales the displayed image without changing either source's resolution.
    Scene USD owns lighting, materials, and environment placement. Newton-model overlays,
    rigid-body dragging, and viewer-only lighting or world offsets are not supported.
    Environment selection controls tiled sensor views; the perspective camera sees the full scene.
    """

    class_type: type[NewtonRTXVisualizer] | str = "{DIR}.newton_visualizer:NewtonRTXVisualizer"
    """Visualizer implementation class."""

    visualizer_type: str = "newton_rtx"
    """Visualizer selector identifier. Do not change."""

    render_settings: dict[str, Any] = dict()
    """RTX attributes to author on the OVRTX render product, as ``{name: (usd_type_name, value)}``.

    ``usd_type_name`` names an ``Sdf.ValueTypeNames`` member, as a string so the config stays
    copyable. For example, ``{"omni:rtx:quality": ("Int", 100)}`` re-enables the path tracer's
    quality convergence loop, disabled by default to keep interactive latency down."""

    cloning_contexts: tuple[type | str, ...] = ("isaaclab_newton.cloner:NewtonReplicateContext",)
    """Build the Newton model supplying poses to the borrowed rendering stage."""

    streaming_view: bool = True
    """Offer compatible scene sensors alongside the interactive RTX perspective camera."""
