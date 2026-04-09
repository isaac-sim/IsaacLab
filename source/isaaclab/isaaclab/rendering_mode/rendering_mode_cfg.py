# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass


@configclass
class RenderingModeCfg:
    """RTX/carb settings for `Omniverse RTX`_ (viewport + RTX camera renderers:
    ``default`` / ``isaac_rtx`` / ``rtx``).

    Put named profiles in :attr:`~isaaclab.sim.SimulationCfg.rendering_mode_cfgs`, then set the same name on
    ``KitVisualizerCfg.rendering_mode`` or ``CameraCfg.renderer_cfg.rendering_mode`` (non-Kit renderers ignore it).

    **Order:** optional :attr:`rendering_mode_preset` (``performance`` / ``balanced`` / ``quality``) loads the
    matching baseline from :mod:`isaaclab.rendering_mode.rendering_mode_presets`—the same three choices as CLI
    ``--rendering_mode``. Each non-``None`` ``kit_*`` field then overrides specific carb paths on top of that
    baseline.

    Baselines before this config are defined by the app experience (e.g. ``apps/isaaclab.python.rendering.kit``,
    ``apps/isaaclab.python.headless.rendering.kit``). Newton / non-Kit viewer options belong on ``NewtonVisualizerCfg``,
    not here.

    .. _Omniverse RTX: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html
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

    kit_disocclusion_scale: float | None = None
    """Scales disocclusion handling for tiled / per-camera rendering (ghosting, newly exposed regions).

    Higher values can reduce disocclusion artifacts at the cost of stability or side effects in some scenes.
    Carb path: ``/rtx/aovConverter/disocclusionScale``.
    """

    kit_nre_compositing_renderer_hints: int | None = None
    """NRE compositing renderer hints (Isaac Lab rendering experiences use ``3``).

    Required for correct compositing when using UsdVol 3D Gaussian content across multiple environments.
    Carb path: ``/omni/rtx/nre/compositing/rendererHints`` (see also
    ``omni.rtx.nre.compositing.rendererHints`` in application ``.kit`` files).
    """

    # TODO: Optional passthrough dict for arbitrary carb keys (cf. legacy RenderCfg.carb_settings).
