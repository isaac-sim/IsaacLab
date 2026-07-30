# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton Warp Renderer."""

from typing import Literal

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils.configclass import configclass


@configclass
class NewtonWarpRendererCfg(RendererCfg):
    """Configuration for Newton Warp Renderer."""

    renderer_type: str = "newton_warp"
    """Type identifier for Newton Warp renderer."""

    enable_textures: bool = True
    """Enable texture-mapped rendering for meshes."""

    enable_shadows: bool = False
    """Enable shadow rays for directional lights."""

    enable_ambient_lighting: bool = True
    """Enable ambient lighting for the scene."""

    enable_backface_culling: bool = True
    """Cull back-facing triangles."""

    max_distance: float = 1000.0
    """Maximum ray distance [m].

    Used as the far clip when a camera does not provide a spawn configuration. When the camera's
    :attr:`~isaaclab.sim.spawners.sensors.PinholeCameraCfg.clipping_range` is available, its far
    value takes precedence per camera (see
    :meth:`~isaaclab_newton.renderers.NewtonWarpRenderer.create_render_data`).
    """

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
    """Clipping behavior for depth values beyond the far plane. Defaults to ``"none"``.

    Newton writes ``0.0`` for rays that miss all geometry or fall beyond the far plane, so the
    background sentinel is ``0.0`` rather than ``+inf`` (unlike the RTX renderer). Consequently:

    - ``"none"``: leave background depth at ``0.0``.
    - ``"zero"``: leave background depth at ``0.0`` (identical to ``"none"`` for this backend).
    - ``"max"``: replace background depth with the far clip [m], mirroring
      :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.depth_clipping_behavior`.
    """

    create_default_light: bool = True
    """Create a default directional light source in the scene."""

    semantic_filter: str | list[str] = "*:*"
    """A string or list specifying a semantic filter predicate for :attr:`semantic_segmentation` and
    :attr:`instance_segmentation`. Defaults to ``"*:*"`` (all semantic types and labels).

    Mirrors :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.semantic_filter`. If a string, it is a
    disjunctive-normal-form predicate over ``(semantic_type, labels)`` clauses separated by ``;``. Each
    clause is ``type:labels`` where ``labels`` supports ``&`` (and), ``|`` / ``,`` (or), ``!`` (not), and
    ``*`` (any); ``type`` may be ``*``. For example ``"class:* ; * : shelf"`` keeps every prim with a
    ``class`` label or any prim carrying the label ``shelf``. If a list, each entry is a bare semantic
    type, i.e. ``["class"]`` is equivalent to ``"class:*"``.

    .. note::
        Semantic labels are read from the USD stage's :class:`UsdSemantics.LabelsAPI` (authored by
        :attr:`~isaaclab.sim.spawners.SpawnerCfg.semantic_tags`), resolving labels inherited from
        ancestor prims. Newton's ray tracer only emits a per-shape index; the semantic/instance
        mappings are reconstructed on the host from ``model.shape_label`` prim paths.
    """

    colorize_semantic_segmentation: bool = True
    """Whether to colorize the semantic segmentation image. Defaults to True.

    If True, :attr:`semantic_segmentation` is returned as an ``(N, H, W, 4) uint8`` RGBA image where each
    semantic id is mapped to a color and ``data.info["semantic_segmentation"]["idToLabels"]`` maps the
    color to its label. If False, it is returned as an ``(N, H, W, 1) int32`` id image and ``idToLabels``
    maps the id to its label.
    """

    colorize_instance_segmentation: bool = True
    """Whether to colorize the semantic instance segmentation image. Defaults to True.

    If True, :attr:`instance_segmentation` is returned as an ``(N, H, W, 4) uint8`` RGBA image, else
    an ``(N, H, W, 1) int32`` id image. ``data.info["instance_segmentation"]`` provides ``idToLabels``
    (id/color to prim path) and ``idToSemantics`` (id/color to semantic label).
    """

    semantic_segmentation_mapping: dict = {}
    """Optional mapping from ``"type:label"`` to an explicit RGBA color tuple for
    :attr:`semantic_segmentation` when :attr:`colorize_semantic_segmentation` is True. Defaults to ``{}``.

    Mirrors :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.semantic_segmentation_mapping`. For
    example ``{"class:cube": (255, 36, 66, 255)}`` forces cube-labeled pixels to that color. Labels absent
    from the mapping fall back to the deterministic palette shared with the RTX renderer.
    """

    render_order: Literal["pixel_priority", "view_priority", "tiled"] = "tiled"
    """Render traversal order for the Newton tiled camera."""

    tile_rendering_width: int = 8
    """Tile width [px] for tiled rendering."""

    tile_rendering_height: int = 8
    """Tile height [px] for tiled rendering."""

    kernel_block_dim: int = 64
    """Thread block dimension forwarded to Newton."""
