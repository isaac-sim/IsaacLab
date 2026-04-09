# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass


@configclass
class RenderingModeCfg:
    """Omniverse RTX rendering controls for Kit profiles (viewport and Kit-style camera renderers).

    These parameters configure the `Omniverse RTX Renderer`_. When you attach a profile to
    :attr:`~isaaclab.sim.SimulationCfg.rendering_mode_cfgs` and reference it from a renderer or
    :class:`~isaaclab_physx.visualizers.kit_visualizer_cfg.KitVisualizerCfg`, Isaac Lab applies the
    corresponding ``kit_*`` fields (and optional :attr:`rendering_mode_preset`) via carb settings.

    Default Kit/RTX behavior for Isaac Lab still comes from the experience files:

    * ``apps/isaaclab.python.rendering.kit`` — simulation with the GUI enabled.
    * ``apps/isaaclab.python.headless.rendering.kit`` — headless simulation.

    Non-``None`` fields here override those defaults for the active profile. Built-in preset names
    (``performance``, ``balanced``, ``quality``) match the baselines in
    :mod:`isaaclab.rendering_mode.rendering_mode_presets`. Choosing a preset via
    :attr:`rendering_mode_preset` or the CLI flag ``--rendering_mode`` behaves like selecting that
    profile for the run.

    .. _Omniverse RTX Renderer: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html
    """

    rendering_mode_preset: Literal["performance", "balanced", "quality"] | None = None
    """Optional built-in RTX baseline (performance, balanced, or quality).

    Values are defined in :mod:`isaaclab.rendering_mode.rendering_mode_presets`. This is the same
    conceptual knob as passing ``--rendering_mode`` to a script: it selects one of the three
    packaged profiles before any per-field ``kit_*`` overrides below are applied.
    """

    kit_enable_translucency: bool | None = None
    """Enables translucency for specular transmissive surfaces such as glass.

    This comes at the cost of some performance. Default in experience files is typically ``False``.
    Carb path: ``/rtx/translucency/enabled``.
    """

    kit_enable_reflections: bool | None = None
    """Enables reflections at the cost of some performance. Default is often ``False`` in presets.

    Carb path: ``/rtx/reflections/enabled``.
    """

    kit_enable_global_illumination: bool | None = None
    """Enables diffuse global illumination at the cost of some performance. Default is often ``False``.

    Carb path: ``/rtx/indirectDiffuse/enabled``.
    """

    kit_antialiasing_mode: Literal["Off", "FXAA", "DLSS", "TAA", "DLAA"] | None = None
    """Anti-aliasing mode (defaults in experience files often favor DLSS where available).

    - **DLSS**: Uses AI to reconstruct higher-resolution frames from a lower-resolution input, using
      motion data and prior-frame feedback.
    - **DLAA**: Uses the same super-resolution stack as DLSS to anti-alias at native resolution for
      maximum image quality.

    Applied through ``omni.replicator.core`` (``set_render_rtx_realtime``) so it stays consistent
    with Replicator; other ``kit_*`` paths are set directly on carb after this call.
    """

    kit_enable_dlssg: bool | None = None
    """Enables DLSS Frame Generation. Default is typically ``False``.

    DLSS-G boosts performance by synthesizing additional frames from sequential frames and motion
    data.

    .. note::

        Requires an Ada Lovelace GPU. Can add thread-related overhead.

    Carb path: ``/rtx-transient/dlssg/enabled``.
    """

    kit_enable_dl_denoiser: bool | None = None
    """Enables the DL denoiser (quality up, performance down).

    Carb path: ``/rtx-transient/dldenoiser/enabled``.
    """

    kit_dlss_mode: Literal[0, 1, 2, 3] | None = None
    """When using DLSS anti-aliasing, selects the performance/quality tradeoff. Default is often ``0``.

    * ``0`` — Performance
    * ``1`` — Balanced
    * ``2`` — Quality
    * ``3`` — Auto

    Carb path: ``/rtx/post/dlss/execMode``.
    """

    kit_enable_direct_lighting: bool | None = None
    """Enables direct light contributions from lights. Default is often ``False`` in tight presets.

    Carb path: ``/rtx/directLighting/enabled``.
    """

    kit_samples_per_pixel: int | None = None
    """Direct lighting samples per pixel (higher improves quality, costs performance). Default is often ``1``.

    Carb path: ``/rtx/directLighting/sampledLighting/samplesPerPixel``.
    """

    kit_enable_shadows: bool | None = None
    """Enables shadows (performance cost). Defaults are often ``True`` when lighting is on.

    Carb path: ``/rtx/shadows/enabled``.
    """

    kit_enable_ambient_occlusion: bool | None = None
    """Enables ambient occlusion at a performance cost. Default is often ``False`` in performance presets.

    Carb path: ``/rtx/ambientOcclusion/enabled``.
    """

    kit_dome_light_upper_lower_strategy: Literal[0, 3, 4] | None = None
    """How the dome light is sampled. Default is often ``0``.

    See `dome light sampling`_ in the Omniverse RTX docs for semantics.

    * ``0`` — **Image-based lighting (IBL)** — Most accurate for high-frequency dome textures; can show
      sampling artifacts in real time.
    * ``3`` — **Limited IBL** — Fastest, least accurate; good when the dome is a minor contributor.
    * ``4`` — **Approximated IBL** — Fast, artifact-free in real time for low-frequency domes; pairs
      with the direct-lighting denoiser.

    Carb path: ``/rtx/domeLight/upperLowerStrategy``.

    .. _dome light sampling: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html#dome-light
    """

    # TODO: Optional passthrough dict for arbitrary carb keys (cf. legacy RenderCfg.carb_settings).
