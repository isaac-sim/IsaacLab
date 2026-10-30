# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for Newton GL and RTX visualizer backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerBackendCfg, VisualizerCfg

if TYPE_CHECKING:
    from .newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer


@configclass
class NewtonGLBackendCfg(VisualizerBackendCfg):
    """Newton OpenGL rendering and model display options."""

    class_type: type[NewtonGLVisualizer] | str = "{DIR}.newton_visualizer:NewtonGLVisualizer"
    visualizer_type: str = "newton_gl"
    cloning_contexts: ClassVar[tuple[type | str, ...]] = ("isaaclab_newton.cloner:NewtonReplicateContext",)

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

    show_particles: bool = True
    """Show simulation particles. Set to False to hide them, for example in a surface-only view."""

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
class NewtonRTXBackendCfg(VisualizerBackendCfg):
    """Simulation-owned OVRTX rendering displayed in a Newton GL window.

    Launch preparation expands perspective sources into ordinary scene CameraCfg entries.
    All views borrow camera output. Scripts without a SceneCfg declare cameras through ReplicateSession,
    construct them with instantiate(camera_cfg), and select them with SceneCameraCfg.
    Window resizing scales the displayed image without changing either source's resolution.
    Scene USD owns lighting, materials, and environment placement. Newton-model overlays,
    rigid-body dragging, and viewer-only lighting or world offsets are not supported.
    Environment selection controls tiled sensor views; the perspective camera sees the full scene.
    """

    class_type: type[NewtonRTXVisualizer] | str = "{DIR}.newton_visualizer:NewtonRTXVisualizer"
    """Visualizer implementation class."""

    launcher_type: ClassVar[str] = "isaaclab_ov.app:OvrtxLauncher"

    visualizer_type: str = "newton_rtx"
    """Visualizer selector identifier. Do not change."""

    render_settings: dict[str, Any] = dict()
    """RTX attributes to author on the OVRTX render product, as ``{name: (usd_type_name, value)}``.

    ``usd_type_name`` names an ``Sdf.ValueTypeNames`` member, as a string so the config stays
    copyable. For example, ``{"omni:rtx:quality": ("Int", 100)}`` re-enables the path tracer's
    quality convergence loop, disabled by default to keep interactive latency down."""

    renderer_cfg: OVRTXRendererCfg = OVRTXRendererCfg()
    """Shared scene renderer. Use the same configuration as scene cameras to share their instance.

    Used when declaring perspective CameraCfg entries. SceneCameraCfg retains its sensor's renderer.
    """


@configclass
class NewtonGLVisualizerCfg(NewtonGLBackendCfg, VisualizerCfg):
    """Deprecated combined configuration; use ``VisualizerCfg(backend=NewtonGLBackendCfg())``.

    Simulation launch translates this declaration before constructing visualizers.
    """

    backend: NewtonGLBackendCfg = NewtonGLBackendCfg()
    streaming_view: bool = True


@configclass
class NewtonRTXVisualizerCfg(NewtonRTXBackendCfg, VisualizerCfg):
    """Deprecated combined configuration; use ``VisualizerCfg(backend=NewtonRTXBackendCfg())``.

    Simulation launch translates this declaration before constructing visualizers.
    """

    backend: NewtonRTXBackendCfg = NewtonRTXBackendCfg()
    streaming_view: bool = True


@configclass
class NewtonVisualizerCfg(NewtonGLVisualizerCfg):
    """Deprecated Newton GL configuration; use ``VisualizerCfg(backend=NewtonGLBackendCfg())``."""
