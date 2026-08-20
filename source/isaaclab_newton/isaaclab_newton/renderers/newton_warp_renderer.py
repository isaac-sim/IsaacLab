# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Warp renderer for tiled camera rendering."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

import newton
import torch
import warp as wp

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp

from ..physics.newton_manager import NewtonManager
from .newton_warp_renderer_cfg import NewtonWarpRendererCfg
from .segmentation import NewtonSegmentationMapper, NewtonSegmentationMapping

if TYPE_CHECKING:
    from isaaclab_ppisp import PpispPipeline

    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

logger = logging.getLogger(__name__)

# Maps ``NewtonWarpRendererCfg.render_order`` strings to Newton's ``RenderOrder`` traversal enum.
_RENDER_ORDER_MAP: dict[str, int] = {
    "pixel_priority": newton.RenderOrder.PIXEL_PRIORITY,
    "view_priority": newton.RenderOrder.VIEW_PRIORITY,
    "tiled": newton.RenderOrder.TILED,
}

_PPISP_IMPORT_ERROR_MESSAGE = (
    "isaaclab_ppisp is required when CameraCfg.isp_cfg is set. "
    "It ships with the Isaac Lab wheel (`pip install isaaclab`); otherwise install the "
    "isaaclab-ppisp extension from the Isaac Lab source checkout."
)


def _raise_missing_ppisp_error(exc: ModuleNotFoundError) -> NoReturn:
    # Only translate missing isaaclab_ppisp imports into the optional-dependency hint;
    # unrelated missing modules should surface unchanged for easier debugging.
    if exc.name != "isaaclab_ppisp" and not (exc.name and exc.name.startswith("isaaclab_ppisp.")):
        raise exc
    raise ModuleNotFoundError(_PPISP_IMPORT_ERROR_MESSAGE, name="isaaclab_ppisp") from exc


class RenderData:
    # Back-compat alias for callers of ``RenderData.OutputNames``.
    OutputNames = RenderBufferKind

    # Maps each supported RenderBufferKind to (CameraOutputs field name, Newton warp dtype).
    # Newton reinterprets the allocated buffer memory: e.g. RGBA is allocated as (N,H,W,4) uint8
    # but Newton's render API consumes it as (world_count,H,W) uint32 (same bytes, packed view).
    #
    # The depth family (``distance_to_camera`` / ``distance_to_image_plane`` / ``depth``) is handled
    # separately in :meth:`set_outputs` rather than through this map, because Newton emits the
    # ray-hit distance (``distance_to_camera``) through ``depth_image`` and the planar depth
    # (``depth`` / ``distance_to_image_plane``) through ``forward_depth_image``.
    #
    # The segmentation family (``semantic_segmentation`` / ``instance_segmentation``) is likewise
    # handled separately: Newton emits a single per-shape index buffer that is remapped into each
    # requested segmentation output by
    # :class:`~isaaclab_newton.renderers.segmentation.NewtonSegmentationMapper`.
    _OUTPUT_MAP: dict[str, tuple[str, type]] = {
        str(RenderBufferKind.RGBA): ("color_image", wp.uint32),
        str(RenderBufferKind.RGB_HDR): ("hdr_color_image", wp.vec3f),
        str(RenderBufferKind.ALBEDO): ("albedo_image", wp.uint32),
        str(RenderBufferKind.NORMALS): ("normals_image", wp.vec3f),
    }

    # Newton's native ``depth_image`` is the ray-hit (euclidean) distance from the camera optical
    # center, which is Isaac Lab's ``distance_to_camera``.
    _RAY_DEPTH_KIND: str = str(RenderBufferKind.DISTANCE_TO_CAMERA)
    # Planar-depth outputs (distance along the camera's forward axis). ``depth`` is Isaac Lab's alias
    # for ``distance_to_image_plane``. Newton fills these directly through ``forward_depth_image``.
    _PLANE_DEPTH_KINDS: frozenset[str] = frozenset(
        {
            str(RenderBufferKind.DEPTH),
            str(RenderBufferKind.DISTANCE_TO_IMAGE_PLANE),
        }
    )

    @dataclass
    class CameraOutputs:
        color_image: wp.array(dtype=wp.uint32, ndim=3) = None
        hdr_color_image: wp.array(dtype=wp.vec3f, ndim=3) = None
        albedo_image: wp.array(dtype=wp.uint32, ndim=3) = None
        # Buffer Newton fills with ray-hit (euclidean) distance; bound to the caller's
        # ``distance_to_camera`` output (``None`` when that output was not requested).
        depth_image: wp.array(dtype=wp.float32, ndim=3) = None
        # Buffer Newton fills with planar (forward-axis) depth; bound to the caller's ``depth`` /
        # ``distance_to_image_plane`` output (``None`` when no planar depth was requested).
        forward_depth_image: wp.array(dtype=wp.float32, ndim=3) = None
        normals_image: wp.array(dtype=wp.vec3f, ndim=3) = None
        # Buffer Newton fills with the per-pixel shape index; the source for all segmentation outputs.
        # Kept 4D ``(world_count, 1, H, W)`` because the segmentation remap kernels index ``[w, c, y, x]``.
        shape_index_image: wp.array(dtype=wp.uint32, ndim=4) = None

    def __init__(
        self,
        render_context: newton.RenderContext,
        spec: CameraRenderSpec,
        seg_mapper: NewtonSegmentationMapper | None = None,
        renderer_cfg: NewtonWarpRendererCfg | None = None,
    ):
        self.render_context = render_context
        # Shared, scene-static segmentation lookup builder (``None`` until segmentation is requested).
        self._seg_mapper = seg_mapper
        self._renderer_cfg = renderer_cfg

        self.num_cameras = 1

        # Per-camera Newton sensor holding the camera-space rays and render settings; created lazily
        # in :meth:`update` once the intrinsics (and therefore the field of view) are known.
        self.sensor_camera: newton.sensors.SensorCamera | None = None
        # Per-world camera-to-world transforms, shape ``(world_count,)``. Updated in place because the
        # sensor manager graph captures the render launch against this buffer.
        self.camera_transforms: wp.array(dtype=wp.transformf) = None
        # Per-world render flags (all ``WorldRenderFlag.ENABLE``); allocated once and reused.
        self.world_render_flags: wp.array(dtype=wp.int32) = None
        self._camera_quat_scratch: wp.array = None
        # Name under which this camera's render launch is registered with the
        # Newton sensor manager (set on first render).
        self.sensor_task_name: str | None = None
        self.outputs = RenderData.CameraOutputs()
        # Requested depth-family destination views keyed by data-type name. Each view aliases the
        # caller's output buffer as ``(world_count, H, W)`` float32.
        self._depth_dests: dict[str, wp.array] = {}
        # Planar-depth destinations in request order. ``depth`` and ``distance_to_image_plane`` are
        # aliases: Newton fills only the first through ``forward_depth_image`` and the rest are copied
        # from it in :meth:`_copy_extra_planar_depth`.
        self._planar_depth_dests: list[wp.array] = []
        # Requested segmentation outputs keyed by data-type name -> (destination view, mapping). Each view
        # aliases the caller's output buffer as ``(world_count, 1, H, W)`` uint32.
        self._seg_dests: dict[str, tuple[wp.array, NewtonSegmentationMapping]] = {}
        self.width = getattr(spec.cfg, "width", 100)
        self.height = getattr(spec.cfg, "height", 100)
        # Camera clipping planes [m] from ``spawn.clipping_range`` (``[0]`` near, ``[1]`` far).
        # Newton's ray tracer has no near-plane parameter, so only the far plane is enforced (through
        # the render config's ``max_distance``); ``near_clip`` is captured for consumers but not applied.
        spawn = getattr(spec.cfg, "spawn", None)
        clipping_range = getattr(spawn, "clipping_range", None)
        self.near_clip: float | None = float(clipping_range[0]) if clipping_range is not None else None
        self.far_clip: float | None = float(clipping_range[1]) if clipping_range is not None else None

        # ABGR clear color packed as uint32 — Newton reads the low byte as R, next as G, next as B,
        # high byte as A (little-endian RGBA in memory). Default is 93% gray (0xFFEEEEEE), matching the
        # RTX renderer background and improving visibility of dark objects.
        background_color = getattr(spec.cfg, "background_color", None)
        if background_color is not None:
            r, g, b = (max(0, min(255, round(c * 255))) for c in background_color)
            self.clear_color: int = (0xFF << 24) | (b << 16) | (g << 8) | r
        else:
            self.clear_color = 0xFFEEEEEE

        # Render settings passed to ``RenderContext.render`` for this camera, and the clear values
        # written to the output buffers before rendering. Built from the renderer cfg (always supplied
        # by ``create_render_data``); left ``None`` only in cfg-less unit fixtures that never render.
        # ``max_distance`` doubles as the far clip, and the per-camera clipping range takes precedence
        # over the renderer default. For ``depth_clipping_behavior == "max"`` the depth background is set
        # to the far clip [m] (mirroring the RTX renderer): Newton writes ``clear_depth`` for rays that
        # miss all geometry or fall beyond ``max_distance``, so both ``distance_to_camera`` and the planar
        # depth outputs get the far clip for background pixels without a post-render pass. Otherwise the
        # background stays at ``0.0``.
        self.render_config: newton.RenderConfig | None = None
        self.clear_data: newton.ClearData | None = None
        if renderer_cfg is not None:
            self.render_config = newton.RenderConfig(
                enable_textures=renderer_cfg.enable_textures,
                enable_shadows=renderer_cfg.enable_shadows,
                enable_ambient_lighting=renderer_cfg.enable_ambient_lighting,
                enable_backface_culling=renderer_cfg.enable_backface_culling,
                max_distance=self.far_clip if self.far_clip is not None else renderer_cfg.max_distance,
                render_order=_RENDER_ORDER_MAP[renderer_cfg.render_order],
                tile_width=renderer_cfg.tile_rendering_width,
                tile_height=renderer_cfg.tile_rendering_height,
            )
            clear_depth = 0.0
            if renderer_cfg.depth_clipping_behavior == "max" and self.far_clip is not None:
                clear_depth = self.far_clip
            self.clear_data = newton.ClearData(clear_color=self.clear_color, clear_depth=clear_depth)

        # Post-render PPISP pipeline composed when ``spec.cfg.isp_cfg`` is set.
        # ``isp_cfg`` is already fully normalized by ``prepare_cameras`` by the time it reaches here.
        self.ppisp_pipeline: PpispPipeline | None = None
        if spec.cfg.isp_cfg is not None:
            try:
                from isaaclab_ppisp import PpispPipeline
            except ModuleNotFoundError as exc:
                _raise_missing_ppisp_error(exc)

            self.ppisp_pipeline = PpispPipeline(spec.cfg.isp_cfg)
        self._hdr_scratch_wp: wp.array | None = None
        """Internal HDR scratch buffer allocated when PPISP is composed but the
        user did not request ``"rgb_hdr"`` in ``data_types``. Also exposed to
        the Newton sensor through :attr:`CameraOutputs.hdr_color_image` as a
        vec3f reinterpretation of this same backing storage."""
        self._ppisp_hdr_source: wp.array | None = None
        """PPISP HDR source bound once in :meth:`set_outputs` from the caller's
        ``rgb_hdr`` output or :attr:`_hdr_scratch_wp`."""
        self._ppisp_rgba_dest: wp.array | None = None
        """PPISP LDR destination bound once in :meth:`set_outputs` from the
        caller's ``rgba`` output."""

    @property
    def model(self) -> newton.Model:
        """The Newton model rendered by this camera's shared render context."""
        return self.render_context.model

    def _render_view(self, proxy: ProxyArray, dtype: type) -> wp.array:
        """Alias the caller's output buffer as a ``(world_count, H, W)`` warp array of ``dtype``.

        Newton reinterprets the backing memory in place (no copy), so ``RenderContext.render`` writes
        directly into the camera's output buffer.
        """
        wp_arr = proxy.warp
        shape = (self.model.world_count, self.height, self.width)
        return wp.array(ptr=wp_arr.ptr, dtype=dtype, shape=shape, device=wp_arr.device, copy=False)

    def _view(self, proxy: ProxyArray, dtype: type, shape: tuple[int, ...]) -> wp.array:
        """Alias the caller's output buffer as a warp array of ``dtype`` with the given ``shape``.

        Used for the segmentation outputs, which the remap kernels consume as ``(world_count, 1, H, W)``.
        """
        wp_arr = proxy.warp
        return wp.array(ptr=wp_arr.ptr, dtype=dtype, shape=shape, device=wp_arr.device, copy=False)

    def set_outputs(self, output_data: dict[str, ProxyArray]):
        model = self.model
        shape_4d = (model.world_count, self.num_cameras, self.height, self.width)
        self._depth_dests = {}
        self._planar_depth_dests = []
        self._seg_dests = {}
        self.outputs.depth_image = None
        self.outputs.forward_depth_image = None
        self.outputs.shape_index_image = None
        for output_name, proxy in output_data.items():
            # Depth family: bind each requested output to a float32 destination view. Newton fills the
            # ray-hit distance through ``depth_image`` and the planar depth through ``forward_depth_image``.
            if output_name == self._RAY_DEPTH_KIND:
                dest = self._render_view(proxy, wp.float32)
                self._depth_dests[output_name] = dest
                self.outputs.depth_image = dest
                continue
            if output_name in self._PLANE_DEPTH_KINDS:
                dest = self._render_view(proxy, wp.float32)
                self._depth_dests[output_name] = dest
                self._planar_depth_dests.append(dest)
                continue
            # Segmentation family: bind each requested output to a destination view — colorized RGBA
            # (uint32 packed) or raw int32 ids (matching the Isaac RTX / OVRTX contract).  Newton
            # fills only the shape-index scratch (uint32), which is remapped into each output in
            # :meth:`_convert_segmentation`.
            if output_name == RenderBufferKind.SEMANTIC_SEGMENTATION:
                colorize = bool(self._renderer_cfg.colorize_semantic_segmentation)
            elif output_name == RenderBufferKind.INSTANCE_SEGMENTATION:
                colorize = bool(self._renderer_cfg.colorize_instance_segmentation)
            else:
                colorize = None
            if colorize is not None:
                if self._seg_mapper is None:
                    raise RuntimeError(
                        f"Output '{output_name}' requires a segmentation mapper, but none was created. "
                        "Ensure the camera's data_types includes the segmentation output."
                    )
                seg_mapping = self._seg_mapper.get_mapping(output_name, colorize)
                dest = self._view(proxy, wp.uint32 if colorize else wp.int32, shape_4d)
                self._seg_dests[output_name] = (dest, seg_mapping)
                continue
            mapping = self._OUTPUT_MAP.get(output_name)
            if mapping is None:
                if output_name != str(RenderBufferKind.RGB):
                    logger.warning(f"NewtonWarpRenderer - output type {output_name} is not yet supported")
                continue
            field_name, dtype = mapping
            setattr(self.outputs, field_name, self._render_view(proxy, dtype))
        # Newton fills only the first planar-depth destination; ``depth`` and ``distance_to_image_plane``
        # are aliases, so any additional planar destinations are copied from it after rendering.
        self.outputs.forward_depth_image = self._planar_depth_dests[0] if self._planar_depth_dests else None
        # Allocate the shape-index buffer Newton fills when any segmentation output is requested; all
        # requested segmentation outputs are remapped from this single buffer in :meth:`_convert_segmentation`.
        if self._seg_dests:
            self.outputs.shape_index_image = wp.zeros(shape_4d, dtype=wp.uint32, device=model.device)
        # When PPISP is composed but the user did not request the raw HDR AOV,
        # allocate an internal HDR scratch buffer and route a vec3f-shaped view
        # of it as the Newton sensor's ``hdr_color_image`` so the renderer
        # fills it directly.
        if self.ppisp_pipeline is not None and self.outputs.hdr_color_image is None:
            ref_proxy = next(iter(output_data.values()))
            self._hdr_scratch_wp = wp.zeros(
                (model.world_count, self.height, self.width, 3),
                dtype=wp.float32,
                device=ref_proxy.device,
            )
            self.outputs.hdr_color_image = wp.array(
                ptr=self._hdr_scratch_wp.ptr,
                dtype=wp.vec3f,
                shape=(model.world_count, self.height, self.width),
                device=self._hdr_scratch_wp.device,
                copy=False,
            )
        # Bind the two warp arrays the per-frame PPISP dispatch needs.
        if self.ppisp_pipeline is not None:
            if str(RenderBufferKind.RGBA) not in output_data:
                raise ValueError(
                    "Newton renderer ISP requires 'rgba' (or 'rgb', which aliases into rgba) as the"
                    " LDR output destination, but neither was provided. Add 'rgb' or 'rgba' to"
                    " Camera.cfg.data_types when isp_cfg is set."
                )
            hdr_proxy = output_data.get(str(RenderBufferKind.RGB_HDR))
            self._ppisp_hdr_source = hdr_proxy.warp if hdr_proxy is not None else self._hdr_scratch_wp
            self._ppisp_rgba_dest = output_data[str(RenderBufferKind.RGBA)].warp

    def get_output(self, output_name: str) -> wp.array:
        if output_name in self._depth_dests:
            return self._depth_dests[output_name]
        elif output_name in self._seg_dests:
            return self._seg_dests[output_name][0]
        elif output_name == RenderBufferKind.RGBA:
            return self.outputs.color_image
        elif output_name == RenderBufferKind.RGB_HDR:
            return self.outputs.hdr_color_image
        elif output_name == RenderBufferKind.ALBEDO:
            return self.outputs.albedo_image
        elif output_name == RenderBufferKind.NORMALS:
            return self.outputs.normals_image
        return None

    def _convert_segmentation(self):
        """Remap Newton's shape-index buffer into each requested segmentation output.

        Newton emits a single per-pixel shape index (:attr:`CameraOutputs.shape_index_image`);
        ``semantic_segmentation`` / ``instance_segmentation`` are each derived from it by a
        :class:`~isaaclab_newton.renderers.segmentation.NewtonSegmentationMapping`.
        No-op when no segmentation output was requested.
        """
        if self.outputs.shape_index_image is None:
            return
        for dest, seg_mapping in self._seg_dests.values():
            seg_mapping.convert_shape_index_to_output(self.outputs.shape_index_image, dest)

    def segmentation_info(self) -> dict[str, dict]:
        """Per-output ``idToLabels`` / ``idToSemantics`` info for the requested segmentation outputs."""
        return {name: seg_mapping.info for name, (_dest, seg_mapping) in self._seg_dests.items()}

    def _copy_extra_planar_depth(self):
        """Fan the rendered planar depth out to any additional planar-depth destinations.

        ``depth`` and ``distance_to_image_plane`` are aliases, so Newton renders only the first planar
        destination (bound to ``forward_depth_image``); the rest share the same values. No-op when at
        most one planar-depth output was requested.
        """
        for dest in self._planar_depth_dests[1:]:
            wp.copy(dest, self._planar_depth_dests[0])

    def update(self, positions: ProxyArray, orientations: ProxyArray, intrinsics: ProxyArray):
        model = self.model
        device = model.device
        # Buffers are persistent: the sensor manager graph captures the render launch against
        # ``camera_transforms``, so it must be updated in place.
        if self._camera_quat_scratch is None:
            self._camera_quat_scratch = wp.empty_like(orientations)
        if self.camera_transforms is None:
            self.camera_transforms = wp.empty(model.world_count, dtype=wp.transformf, device=device)
        if self.world_render_flags is None:
            self.world_render_flags = wp.full(
                model.world_count,
                value=int(newton.WorldRenderFlag.ENABLE),
                dtype=wp.int32,
                device=device,
            )
        converted_wp = self._camera_quat_scratch
        convert_camera_frame_orientation_convention_wp(
            src=orientations,
            dst=converted_wp,
            origin="world",
            target="opengl",
            device=device,
        )

        wp.launch(
            RenderData._update_transforms,
            model.world_count,
            [positions, converted_wp, self.camera_transforms],
            device=device,
        )

        if self.sensor_camera is None:
            # Newton derives a single vertical field of view from fy; ``compute_camera_rays_pinhole``
            # takes a scalar fov and returns camera-space rays of shape (H, W, 2).
            first_focal_length = intrinsics.torch[:, 1, 1][0:1]
            fov_radians = float(2.0 * torch.atan(self.height / (2.0 * first_focal_length)))
            rays = newton.sensors.SensorCamera.compute_camera_rays_pinhole(
                self.width, self.height, camera_fov=fov_radians, device=device
            )
            self.sensor_camera = newton.sensors.SensorCamera(rays, self.render_context)
            self.sensor_camera.render_config = self.render_config
            self.sensor_camera.clear_data = self.clear_data

    @wp.kernel
    def _update_transforms(
        positions: wp.array(dtype=wp.vec3f),
        orientations: wp.array(dtype=wp.quatf),
        output: wp.array(dtype=wp.transformf),
    ):
        tid = wp.tid()
        output[tid] = wp.transformf(positions[tid], orientations[tid])


class NewtonWarpRenderer(BaseRenderer):
    """Newton Warp backend for tiled camera rendering."""

    RenderData = RenderData

    def __init__(self, cfg: NewtonWarpRendererCfg):
        """Pre-physics initialization."""
        from isaaclab.physics.scene_data_requirements import (
            aggregate_requirements,
            requirement_for_renderer_type,
        )

        self.cfg = cfg
        # Shared render engine constructed post-physics from the built Newton model; drives every
        # camera's ``RenderContext.render`` call.
        self.render_context: newton.RenderContext | None = None
        # USD stage captured in ``prepare_cameras``; used by the segmentation mapper to read semantics.
        self._stage: Any = None
        # Shared, scene-static segmentation lookup builder, created lazily in ``create_render_data``.
        self._seg_mapper: NewtonSegmentationMapper | None = None

        sim = SimulationContext.instance()
        current_req = sim.get_scene_data_requirements()
        renderer_req = requirement_for_renderer_type("newton_warp")
        merged = aggregate_requirements([current_req, renderer_req])
        if merged != current_req:
            sim.update_scene_data_requirements(merged)

    def initialize(self) -> None:
        """Post-physics setup: read the built Newton model and construct the render context."""
        self._newton_model = NewtonManager.get_model()
        if self._newton_model is None:
            raise RuntimeError(
                "NewtonWarpRenderer requires a Newton model but the Newton manager has no model. "
                "This usually means the Newton model failed to build from the USD stage "
                "(e.g., unsupported PhysX schemas such as tendons). "
                "Check the log for earlier Newton model build errors."
            )

        from vulkan_renderer import NewtonAdapter
        # Textures are only sampled when ``enable_textures`` is set, so skip loading them otherwise.
        self.render_context = NewtonAdapter(self._newton_model)

        # if self.cfg.create_default_light:
        #     self.render_context.create_default_light(enable_shadows=self.cfg.enable_shadows)

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output layout this Newton Warp backend writes.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.supported_output_types`."""

        def seg_spec(colorize: bool) -> RenderBufferSpec:
            # Colorized segmentation is RGBA uint8; raw segmentation is a single int32 id channel
            # (matching the Isaac RTX / OVRTX contract so backend-independent consumers see the same dtype).
            return RenderBufferSpec(4, wp.uint8) if colorize else RenderBufferSpec(1, wp.int32)

        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.ALBEDO: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.SEMANTIC_SEGMENTATION: seg_spec(self.cfg.colorize_semantic_segmentation),
            RenderBufferKind.INSTANCE_SEGMENTATION: seg_spec(self.cfg.colorize_instance_segmentation),
        }

    def prepare_cameras(self, stage: Any, spec: CameraRenderSpec) -> None:
        """Resolve the camera's PPISP cfg before rendering.

        :mod:`isaaclab.sensors.camera` does not depend on PPISP; the renderer
        owns the sentinel-resolution + cfg-normalization step. Newton has no
        USD-side overrides to author beyond this.

        Also captures the USD ``stage`` so the segmentation mapper can read the scene's
        :class:`UsdSemantics.LabelsAPI` labels when a segmentation output is requested.
        """
        self._stage = stage
        # NOTE: OpenCV lens distortion (``spawn.distortion``) is not yet applied by the Newton
        # renderer. The distortion cfg is renderer-agnostic and could be piped through Newton's warp
        # ray-tracing utilities here in the future; for now the camera renders undistorted. This is
        # the intended extension point.
        spawn = getattr(spec.cfg, "spawn", None)
        if getattr(spawn, "distortion", None) is not None:
            logger.warning(
                "OpenCV lens distortion is set on the camera cfg but is not yet applied by the Newton"
                " renderer: it derives a single field of view from fy, so the distortion coefficients,"
                " the principal point, and a non-square fx are ignored and the camera renders as a"
                " centered, square-pixel pinhole. Use the RTX/OVRTX renderer to apply the full model."
            )
        if spec.cfg.isp_cfg is None:
            return
        try:
            from isaaclab_ppisp import resolve_and_normalize
        except ModuleNotFoundError as exc:
            _raise_missing_ppisp_error(exc)

        camera_prim_path = spec.camera_prim_paths[0] if spec.camera_prim_paths else None
        spec.cfg.isp_cfg = resolve_and_normalize(spec.cfg.isp_cfg, stage, camera_prim_path)

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """No-op for Newton Warp - uses Newton scene directly without stage export.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage`."""
        pass

    def create_render_data(self, spec: CameraRenderSpec) -> RenderData:
        """Create render data for the Newton camera.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`."""

        # Build the shared segmentation mapper and its per-kind lookup tables up-front for all
        # requested segmentation outputs.
        if (
            RenderBufferKind.SEMANTIC_SEGMENTATION in spec.cfg.data_types
            or RenderBufferKind.INSTANCE_SEGMENTATION in spec.cfg.data_types
        ):
            if self._seg_mapper is None:
                self._seg_mapper = NewtonSegmentationMapper(self._newton_model, self._stage, self.cfg)
        if RenderBufferKind.SEMANTIC_SEGMENTATION in spec.cfg.data_types:
            self._seg_mapper.build_mapping(
                RenderBufferKind.SEMANTIC_SEGMENTATION, bool(self.cfg.colorize_semantic_segmentation)
            )
        if RenderBufferKind.INSTANCE_SEGMENTATION in spec.cfg.data_types:
            self._seg_mapper.build_mapping(
                RenderBufferKind.INSTANCE_SEGMENTATION, bool(self.cfg.colorize_instance_segmentation)
            )

        render_data = RenderData(self.render_context, spec, seg_mapper=self._seg_mapper, renderer_cfg=self.cfg)
        return render_data

    def set_outputs(self, render_data: RenderData, output_data: dict[str, ProxyArray]):
        """Store output buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`."""
        render_data.set_outputs(output_data)

    def update_transforms(self):
        """Sync Newton scene state before rendering.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_transforms`."""
        sim = SimulationContext.instance()
        sim.physics_manager.forward()
        NewtonManager.update_visualization_state()

    def update_geometries(self) -> None:
        """No-op for Newton Warp - geometry is read directly from Newton state during render.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_geometries`."""
        pass

    def update_camera(
        self,
        render_data: RenderData,
        positions: ProxyArray,
        orientations: ProxyArray,
        intrinsics: ProxyArray,
    ):
        """Update camera poses and intrinsics.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_camera`."""
        render_data.update(positions, orientations, intrinsics)

    def render(self, render_data: RenderData):
        """Render and write to output buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.render`."""

        # Refresh the shadow state under PhysX before the manager refits the BVH.
        NewtonManager.get_state()
        if render_data.sensor_task_name is None:
            render_data.sensor_task_name = f"newton_warp_render:{id(render_data)}"
            # The Vulkan render signals a CUDA external semaphore, which a conditional CUDA graph
            # body forbids; register it as non-capturable so it runs eagerly and leaves the main
            # solver and ray-cast sensor graphs captured.
            NewtonManager._register_sensor_task(
                render_data.sensor_task_name,
                lambda: self._launch_render(render_data),
                graph_capturable=False,
            )
        NewtonManager._update_sensor_tasks(render_data.sensor_task_name)

        # Post-render PPISP: HDR scene-linear → LDR RGBA. Source/destination
        # tensors were bound once in ``set_outputs``.
        if render_data.ppisp_pipeline is not None:
            render_data.ppisp_pipeline.apply(
                render_data._ppisp_hdr_source,
                render_data._ppisp_rgba_dest,
            )

    def _launch_render(self, render_data: RenderData) -> None:
        """Launch the camera render kernels for sensor graph capture.

        Isaac Lab drives camera poses externally (through :meth:`update_camera`), so the render goes
        through the shared :class:`~newton.RenderContext` with the pre-computed per-world transforms
        rather than through :meth:`newton.sensors.SensorCamera.update`, which reads its transforms from
        a model site. The per-camera :class:`~newton.sensors.SensorCamera` supplies the rays and the
        render/clear settings.
        """
        sensor_camera = render_data.sensor_camera
        world_count = render_data.model.world_count

        # Sync triangle-mesh (deformable) points from the current state; a no-op for rigid-only scenes.
        state = NewtonManager.get_state_0()
        self.render_context.update(state)

        self.render_context.render(
            state,
            camera_transforms=render_data.camera_transforms,
            camera_rays=sensor_camera.rays,
            world_render_flags=render_data.world_render_flags,
            color_image=render_data.outputs.color_image,
            hdr_color_image=render_data.outputs.hdr_color_image,
            albedo_image=render_data.outputs.albedo_image,
            depth_image=render_data.outputs.depth_image,
            forward_depth_image=render_data.outputs.forward_depth_image,
            normal_image=render_data.outputs.normals_image,
            shape_index_image=(
                render_data.outputs.shape_index_image.reshape((world_count, render_data.height, render_data.width))
                if render_data.outputs.shape_index_image is not None
                else None
            ),
            clear_data=sensor_camera.clear_data,
            config=sensor_camera.render_config,
            kernel_block_dim=self.cfg.kernel_block_dim,
        )

        # ``depth`` and ``distance_to_image_plane`` are aliases; fan the rendered planar depth out to
        # any additional planar-depth destinations.
        render_data._copy_extra_planar_depth()

        # Remap the shape-index buffer into the requested segmentation outputs.
        render_data._convert_segmentation()

    def read_output(self, render_data: RenderData, camera_data: CameraData) -> None:
        """Copy rendered outputs to the camera data buffers.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.read_output`."""
        for output_name in camera_data.output:
            if output_name == "rgb":
                continue
            image_data = render_data.get_output(output_name)
            if image_data is not None:
                output_wp = camera_data.output[output_name].warp
                if image_data.ptr != output_wp.ptr:
                    wp.copy(output_wp, image_data)

        # Publish the segmentation id-to-label metadata (idToLabels / idToSemantics) alongside the
        # pixel buffers.
        for output_name, info in render_data.segmentation_info().items():
            camera_data.info[output_name] = info

    def cleanup(self, render_data: RenderData | None):
        """Release resources and drop the camera's sensor task.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""
        if render_data:
            if render_data.sensor_task_name is not None:
                NewtonManager._unregister_sensor_task(render_data.sensor_task_name)
                render_data.sensor_task_name = None
            render_data.sensor_camera = None
