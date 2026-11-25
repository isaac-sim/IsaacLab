# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for OVRTX Renderer."""

from __future__ import annotations

import os
import tempfile
from dataclasses import MISSING
from typing import TYPE_CHECKING, ClassVar, Literal

import warp as wp

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .ovrtx_renderer import OVRTXBackend, OVRTXRenderer


@configclass
class OVRTXBackendCfg(BackendCfg):
    """Native engine settings and the configuration identifying its scene resource."""

    class_type: type[OVRTXBackend] | str = "{DIR}.ovrtx_renderer:OVRTXBackend"
    scene_key: OVRTXRendererCfg | None = None
    """None shares the simulation scene; standalone renderers retain their own scene policy."""
    log_file_path: str = os.path.join(tempfile.gettempdir(), "ovrtx_renderer.log")
    """Native renderer log destination."""
    # FIXME: revert once OVRTX stops warning on every cloned attribute without data.
    log_level: str = "error"
    """Native renderer log level."""
    use_ovstage: bool = MISSING
    """Whether the engine borrows a simulation-owned OVStage instead of using its internal stage."""
    read_gpu_transforms: bool = MISSING
    """Whether OVRTX reads its GPU transform cache, resolved before native construction."""
    device: str = MISSING
    """Normalized simulation device: ``cpu`` or ``cuda:<index>``."""
    motion_bvh: Literal["disable", "enable", "auto"] | None = None
    """Motion BVH mode passed to the native engine."""


@configclass
class OVRTXRendererCfg(RendererCfg):
    """Configuration for OVRTX Renderer.

    The OVRTX renderer uses the ovrtx library for high-fidelity RTX-based rendering.
    width, height, num_envs, and data_types are obtained from the
    :class:`~isaaclab.renderers.camera_render_spec.CameraRenderSpec` when
    :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data` is called
    (same pattern as Isaac RTX).
    """

    class_type: type[OVRTXRenderer] | str = "{DIR}.ovrtx_renderer:OVRTXRenderer"
    """Renderer implementation class."""

    renderer_type: str = "ovrtx"
    """Type identifier for OVRTX renderer."""

    launcher_type: ClassVar[str] = "isaaclab_ov.app:OvrtxLauncher"
    """The launcher that registers the OVRTX runtime before the stage is created."""

    cloning_contexts: tuple[type | str, ...] = ("isaaclab_ov.cloner:OvrtxReplicateContext",)
    """Request native OVRTX copies; clone preparation may select a shared OVStage instead."""

    async_rendering: bool = False
    """Trade one capture of camera latency for pipelined rendering. Defaults to False (synchronous).

    Each camera returns its previous capture while rendering its next image. The first capture
    and the first capture after reset wait for a fresh image. Live pose and calibration fields
    stay current. ``camera.data.info[output_name]["capture"]`` carries the image's matching
    ``pos_w``, ``quat_w_world``, ``intrinsic_matrices``, and per-environment ``frame`` indices.
    The ovstage path does not support pipelining and renders synchronously.
    """

    max_envs_per_render_product: int | Literal["auto", "all"] = "auto"
    """Maximum number of environments in one render product of a camera sensor. Defaults to ``"auto"``.

    A render product renders the images of its environments into one tiled image. A very large camera sensor can
    fail to render with one render product. Several render products, each with a smaller tiled image, can avoid this
    failure. The output shapes and the environment order do not change.

    - ``"auto"``: A camera sensor starts with one render product. If its first frame fails, the renderer searches
      for a split. A warning then gives the integer of the split.
    - ``"all"``: A camera sensor always has one render product.
    - An integer ``M`` of at least 1: A camera sensor always has ``ceil(num_envs / M)`` render products. This skips
      the search.

    The value applies to every camera sensor of this renderer configuration, but each failed frame of the search
    splits only one sensor. Set the same value on every OVRTX renderer configuration of a task. Without a shared
    OVStage, another value gives another OVRTX renderer with its own copy of the scene. See
    :ref:`renderers-ovrtx-render-product-split`.
    """

    temp_usd_dir: str | None = None
    """Directory for temporary USD debug dumps written during OVRTX stage preparation.

    When set, the renderer writes ``pre_ovrtx_renderer_stage.usda`` (raw stage before
    partition attributes and export trimming) and ``ovrtx_renderer_stage.usda`` (exported
    stage plus injected render products) under this directory. Must be writable.
    """

    # FIXME: revert once OVRTX stops warning on every cloned attribute without data.
    log_level: str = "error"
    """OVRTX carb log level: "verbose", "info", "warn", "error"."""

    log_file_path: str = os.path.join(tempfile.gettempdir(), "ovrtx_renderer.log")
    """Path for OVRTX log file. Defaults to ``<system temp>/ovrtx_renderer.log``."""

    enable_shadows: bool = False
    """Whether lights cast shadows in RTX Minimal mode. Defaults to False.

    Shadow rays cost render time that rarely changes what a policy learns, so they are turned off
    and opted back into for visually faithful renders.

    Only the ``simple_shading_constant_diffuse``, ``simple_shading_diffuse_mdl`` and
    ``simple_shading_full_mdl`` data types are affected, because those are the ones that put the
    render product into RTX Minimal mode, whose ``omni:rtx:minimal:castShadows`` switch this drives.
    OVRTX's path-traced modes offer no equivalent switch and always cast shadows, so this setting
    does not change ``rgb`` and the other AOV outputs.
    """

    colorize_semantic_segmentation: bool = True
    """Whether to colorize semantic segmentation output. Defaults to True.

    If True, semantic IDs are mapped to RGBA colors and returned as a ``uint8`` 4-channel array.
    If False, raw semantic IDs are returned as an ``int32`` 1-channel array.

    Regardless of this setting, the semantic ID (or color) to label mapping is exposed via
    ``camera.data.info["semantic_segmentation"]["idToLabels"]``.
    """

    colorize_instance_segmentation: bool = True
    """Whether to colorize instance segmentation output. Defaults to True.

    If True, instance IDs are mapped to RGBA colors and returned as a ``uint8`` 4-channel array.
    If False, raw instance IDs are returned as an ``int32`` 1-channel array.

    Regardless of this setting, the instance ID (or color) to prim-path mapping is exposed via
    ``camera.data.info["instance_segmentation"]["idToLabels"]`` and the instance ID (or color) to
    semantic-label mapping via ``camera.data.info["instance_segmentation"]["idToSemantics"]``.
    """

    semantic_segmentation_mapping: dict[str, tuple[int, int, int, int]] = {}
    """RGBA colors for semantic labels such as ``{"class:robot": (61, 178, 255, 255)}``.

    Unspecified labels keep their generated colors. Applies to colorized semantic segmentation
    pixels and their ``idToLabels`` metadata, matching the Isaac RTX palette contract.
    """

    colorize_instance_id_segmentation: bool = True
    """Retained compatibility field; ignored by the OVRTX renderer. Defaults to True.

    OVRTX does not support ``instance_id_segmentation_fast``. Use
    :attr:`colorize_instance_segmentation` to control colorization of the supported instance segmentation output.
    """

    motion_bvh: Literal["disable", "enable", "auto"] | None = None
    """Motion BVH mode for non-visual sensor pipelines."""

    def validate_config(self) -> None:
        """Validate the render product split."""
        max_envs = self.max_envs_per_render_product
        if not (type(max_envs) is int and max_envs >= 1) and max_envs not in ("auto", "all"):
            raise ValueError(
                'OVRTXRendererCfg.max_envs_per_render_product must be "auto", "all" or an integer >= 1,'
                f" got {max_envs!r}."
            )
        for label, color in self.semantic_segmentation_mapping.items():
            if len(color) != 4 or any(type(channel) is not int or not 0 <= channel <= 255 for channel in color):
                raise ValueError(f"Semantic color for {label!r} must have four integer channels in [0, 255].")

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Return the per-output layouts supported by the OVRTX renderer."""

        def segmentation_spec(colorize: bool) -> RenderBufferSpec:
            return RenderBufferSpec(4, wp.uint8) if colorize else RenderBufferSpec(1, wp.int32)

        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.ALBEDO: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_FULL_MDL: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SEMANTIC_SEGMENTATION: segmentation_spec(self.colorize_semantic_segmentation),
            RenderBufferKind.INSTANCE_SEGMENTATION: segmentation_spec(self.colorize_instance_segmentation),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.MOTION_VECTORS: RenderBufferSpec(2, wp.float32),
        }
