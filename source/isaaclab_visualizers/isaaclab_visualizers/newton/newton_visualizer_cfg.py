# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for the Newton visualizer backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerBackendCfg, VisualizerCfg

if TYPE_CHECKING:
    from .newton_visualizer import NewtonGLVisualizer


@configclass
class NewtonRecordingCfg:
    """Output settings of the Recording section in the Newton GL HUD.

    Every output lands in ``<output_dir>/<kind>/``, where ``kind`` is ``video``, ``scene``, or ``checkpoint``.
    The HUD edits a copy of these values while the window is open.
    """

    output_dir: str = "recording"
    """Root folder of all recordings. Relative paths resolve against the working directory, which is the repository
    folder when the run starts there."""

    filename_prefix: str = "newton"
    """Start of every recording's file name."""

    auto_increment: bool = True
    """Append a counter (``_0001``, ``_0002``, ...) so a new recording never overwrites an earlier one."""

    timestamp: bool = False
    """Append the start date and time (``YYYYmmdd-HHMMSS``) to the file name."""

    video_fps: int | None = None
    """Frame rate of the encoded video [Hz]. ``None`` uses the window's update rate (:attr:`WindowCfg.fps`)."""

    video_length: float | None = None
    """Duration of the video [s] after which recording stops by itself. ``None`` records until stopped."""


@configclass
class NewtonGLBackendCfg(VisualizerBackendCfg):
    """Newton visualizer options: OpenGL rendering and model display, and the optional OVRTX view."""

    class_type: type[NewtonGLVisualizer] | str = "{DIR}.newton_visualizer:NewtonGLVisualizer"
    visualizer_type: str = "newton"
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

    recording: NewtonRecordingCfg = NewtonRecordingCfg()
    """Defaults of the HUD's Recording section."""

    rtx_camera: bool | None = None
    """Declare an OVRTX viewer camera so the HUD can switch the main view between Newton GL and OVRTX.

    The camera is added to the scene at launch because scene sensors cannot be added later. It renders only
    while the OVRTX view is selected. ``None`` enables it when the ``ovrtx`` package is installed and the physics
    backend is not PhysX; ``False`` disables it.
    """

    initial_renderer: Literal["opengl", "rtx"] = "opengl"
    """Renderer the window opens with: the Newton OpenGL view or the OVRTX viewer camera.

    ``"rtx"`` requires the OVRTX camera (see :attr:`rtx_camera`) and a kitless runtime, and starts in the RTX view
    that the HUD's Renderer selector otherwise switches to.
    """

    rtx_renderer_cfg: OVRTXRendererCfg = OVRTXRendererCfg()
    """Renderer of the OVRTX viewer camera."""

    rtx_render_settings: dict[str, Any] = dict()
    """Additional RTX render settings of the OVRTX viewer camera."""


@configclass
class NewtonGLVisualizerCfg(NewtonGLBackendCfg, VisualizerCfg):
    """Deprecated combined configuration; use ``VisualizerCfg(backend=NewtonGLBackendCfg())``.

    Simulation launch translates this declaration before constructing visualizers.
    """

    backend: NewtonGLBackendCfg = NewtonGLBackendCfg()
    streaming_view: bool = True


@configclass
class NewtonVisualizerCfg(NewtonGLVisualizerCfg):
    """Deprecated Newton GL configuration; use ``VisualizerCfg(backend=NewtonGLBackendCfg())``."""
