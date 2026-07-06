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
from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp, replace_background_depth_wp

from ..physics.newton_manager import NewtonManager
from .newton_warp_renderer_cfg import NewtonWarpRendererCfg
from .segmentation import SegmentationMapper, SegmentationPlan

if TYPE_CHECKING:
    from isaaclab_ppisp import PpispPipeline

    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

logger = logging.getLogger(__name__)

_PPISP_IMPORT_ERROR_MESSAGE = (
    "isaaclab_ppisp is required when CameraCfg.isp_cfg is set. "
    "Install Isaac Lab with the 'all' extra (`pip install isaaclab[all]`) or install the "
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
    # but the Newton sensor API consumes it as (world_count,1,H,W) uint32 (same bytes, packed view).
    #
    # The depth family (``distance_to_camera`` / ``distance_to_image_plane`` / ``depth``) is handled
    # separately in :meth:`set_outputs` rather than through this map, because Newton emits a single
    # ray-hit-distance buffer that must be reused as the source for the planar-depth conversion.
    #
    # The segmentation family (``semantic_segmentation`` / ``instance_segmentation_fast`` /
    # ``instance_id_segmentation_fast``) is likewise handled separately: Newton emits a single
    # per-shape index buffer that is remapped into each requested segmentation output by
    # :class:`~isaaclab_newton.renderers.segmentation.SegmentationMapper`.
    _OUTPUT_MAP: dict[str, tuple[str, type]] = {
        str(RenderBufferKind.RGBA): ("color_image", wp.uint32),
        str(RenderBufferKind.RGB_HDR): ("hdr_color_image", wp.vec3f),
        str(RenderBufferKind.ALBEDO): ("albedo_image", wp.uint32),
        str(RenderBufferKind.NORMALS): ("normals_image", wp.vec3f),
    }

    # Segmentation outputs, and the renderer cfg flag selecting colorized (uint8 RGBA) vs raw (int32)
    # layout for each. All three are derived from Newton's single ``shape_index_image`` buffer.
    _SEG_COLORIZE_ATTR: dict[str, str] = {
        str(RenderBufferKind.SEMANTIC_SEGMENTATION): "colorize_semantic_segmentation",
        str(RenderBufferKind.INSTANCE_SEGMENTATION_FAST): "colorize_instance_segmentation",
        str(RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST): "colorize_instance_id_segmentation",
    }

    # Newton's native ``depth_image`` is the ray-hit (euclidean) distance from the camera optical
    # center, which is Isaac Lab's ``distance_to_camera``.
    _RAY_DEPTH_KIND: str = str(RenderBufferKind.DISTANCE_TO_CAMERA)
    # Planar-depth outputs (distance along the camera's forward axis). ``depth`` is Isaac Lab's alias
    # for ``distance_to_image_plane``. Both are derived from the ray depth via
    # ``convert_ray_depth_to_forward_depth``.
    _PLANE_DEPTH_KINDS: frozenset[str] = frozenset(
        {
            str(RenderBufferKind.DEPTH),
            str(RenderBufferKind.DISTANCE_TO_IMAGE_PLANE),
        }
    )

    @dataclass
    class CameraOutputs:
        color_image: wp.array(dtype=wp.uint32, ndim=4) = None
        hdr_color_image: wp.array(dtype=wp.vec3f, ndim=4) = None
        albedo_image: wp.array(dtype=wp.uint32, ndim=4) = None
        # Buffer Newton fills with ray-hit (euclidean) distance. Bound either to the caller's
        # ``distance_to_camera`` output or to an internal scratch buffer (see :meth:`set_outputs`).
        depth_image: wp.array(dtype=wp.float32, ndim=4) = None
        normals_image: wp.array(dtype=wp.vec3f, ndim=4) = None
        # Buffer Newton fills with the per-pixel shape index; the source for all segmentation outputs.
        shape_index_image: wp.array(dtype=wp.uint32, ndim=4) = None

    def __init__(
        self,
        newton_sensor: newton.sensors.SensorTiledCamera,
        spec: CameraRenderSpec,
        seg_mapper: SegmentationMapper | None = None,
        renderer_cfg: NewtonWarpRendererCfg | None = None,
    ):
        self.newton_sensor = newton_sensor
        # Shared, scene-static segmentation lookup builder (``None`` until segmentation is requested).
        self._seg_mapper = seg_mapper
        self._renderer_cfg = renderer_cfg

        self.num_cameras = 1

        self.camera_rays: wp.array(dtype=wp.vec3f, ndim=4) = None
        self.camera_transforms: wp.array(dtype=wp.transformf, ndim=2) = None
        self.outputs = RenderData.CameraOutputs()
        # Requested depth-family destination views keyed by data-type name. Each view aliases the
        # caller's output buffer as ``(world_count, 1, H, W)`` float32.
        self._depth_dests: dict[str, wp.array] = {}
        # Internal ray-depth buffer allocated only when a planar-depth output is requested without
        # ``distance_to_camera``; gives ``convert_ray_depth_to_forward_depth`` a source to read from.
        self._ray_depth_scratch: wp.array | None = None
        # Requested segmentation outputs keyed by data-type name -> (destination view, plan). Each view
        # aliases the caller's output buffer as ``(world_count, 1, H, W)`` uint32 (colorized) or int32.
        self._seg_dests: dict[str, tuple[wp.array, SegmentationPlan]] = {}
        # Internal buffer Newton fills with the per-pixel shape index, shared by all segmentation
        # outputs; allocated only when at least one segmentation output is requested.
        self._shape_index_scratch: wp.array | None = None
        self.width = getattr(spec.cfg, "width", 100)
        self.height = getattr(spec.cfg, "height", 100)
        # Camera clipping planes [m] from ``spawn.clipping_range`` (``[0]`` near, ``[1]`` far).
        # Newton's ray tracer has no near-plane parameter, so only the far plane is enforced (through
        # the sensor's ``max_distance``); ``near_clip`` is captured for consumers but not applied.
        spawn = getattr(spec.cfg, "spawn", None)
        clipping_range = getattr(spawn, "clipping_range", None)
        self.near_clip: float | None = float(clipping_range[0]) if clipping_range is not None else None
        self.far_clip: float | None = float(clipping_range[1]) if clipping_range is not None else None
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

    def _view(self, proxy: ProxyArray, dtype: type, shape: tuple[int, ...]) -> wp.array:
        """Alias the caller's output buffer as a ``(world_count, 1, H, W)`` warp array of ``dtype``.

        Newton reinterprets the backing memory in place (no copy), so the sensor writes directly
        into the camera's output buffer.
        """
        wp_arr = proxy.warp
        return wp.array(ptr=wp_arr.ptr, dtype=dtype, shape=shape, device=wp_arr.device, copy=False)

    def set_outputs(self, output_data: dict[str, ProxyArray]):
        shape = (self.newton_sensor.model.world_count, self.num_cameras, self.height, self.width)
        self._depth_dests = {}
        self._ray_depth_scratch = None
        self._seg_dests = {}
        self._shape_index_scratch = None
        ray_depth_dest: wp.array | None = None
        for output_name, proxy in output_data.items():
            # Depth family: bind each requested output to a float32 destination view. Newton fills
            # only the ray-hit distance; planar outputs are derived from it in :meth:`convert_plane_depth`.
            if output_name == self._RAY_DEPTH_KIND or output_name in self._PLANE_DEPTH_KINDS:
                dest = self._view(proxy, wp.float32, shape)
                self._depth_dests[output_name] = dest
                if output_name == self._RAY_DEPTH_KIND:
                    ray_depth_dest = dest
                continue
            # Segmentation family: bind each requested output to a uint32 (colorized RGBA) or int32
            # (raw ids) destination view; Newton fills only the shape-index scratch, which is remapped
            # into each output in :meth:`convert_segmentation`.
            if output_name in self._SEG_COLORIZE_ATTR:
                colorize = bool(getattr(self._renderer_cfg, self._SEG_COLORIZE_ATTR[output_name]))
                plan = self._seg_mapper.plan(output_name, colorize)
                dest = self._view(proxy, wp.uint32 if colorize else wp.int32, shape)
                self._seg_dests[output_name] = (dest, plan)
                continue
            mapping = self._OUTPUT_MAP.get(output_name)
            if mapping is None:
                if output_name != str(RenderBufferKind.RGB):
                    logger.warning(f"NewtonWarpRenderer - output type {output_name} is not yet supported")
                continue
            field_name, dtype = mapping
            setattr(self.outputs, field_name, self._view(proxy, dtype, shape))
        # Bind the buffer Newton fills with ray-hit distance. Write straight into the
        # ``distance_to_camera`` output when requested; otherwise allocate an internal scratch so the
        # planar-depth conversion has a source to read from.
        if ray_depth_dest is not None:
            self.outputs.depth_image = ray_depth_dest
        elif any(name in self._PLANE_DEPTH_KINDS for name in self._depth_dests):
            self._ray_depth_scratch = wp.zeros(shape, dtype=wp.float32, device=self.newton_sensor.model.device)
            self.outputs.depth_image = self._ray_depth_scratch
        else:
            self.outputs.depth_image = None
        # Allocate the shape-index buffer Newton fills when any segmentation output is requested; all
        # requested segmentation outputs are remapped from this single buffer in :meth:`convert_segmentation`.
        if self._seg_dests:
            self._shape_index_scratch = wp.zeros(shape, dtype=wp.uint32, device=self.newton_sensor.model.device)
        self.outputs.shape_index_image = self._shape_index_scratch
        # When PPISP is composed but the user did not request the raw HDR AOV,
        # allocate an internal HDR scratch buffer and route a vec3f-shaped view
        # of it as the Newton sensor's ``hdr_color_image`` so the renderer
        # fills it directly.
        if self.ppisp_pipeline is not None and self.outputs.hdr_color_image is None:
            ref_proxy = next(iter(output_data.values()))
            self._hdr_scratch_wp = wp.zeros(
                (self.newton_sensor.model.world_count, self.height, self.width, 3),
                dtype=wp.float32,
                device=ref_proxy.device,
            )
            self.outputs.hdr_color_image = wp.array(
                ptr=self._hdr_scratch_wp.ptr,
                dtype=wp.vec3f,
                shape=shape,
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

    def convert_segmentation(self):
        """Remap Newton's shape-index buffer into each requested segmentation output.

        Newton emits a single per-pixel shape index (:attr:`CameraOutputs.shape_index_image`);
        ``semantic_segmentation`` / ``instance_segmentation_fast`` / ``instance_id_segmentation_fast``
        are each derived from it by a :class:`~isaaclab_newton.renderers.segmentation.SegmentationPlan`.
        No-op when no segmentation output was requested.
        """
        if self._shape_index_scratch is None:
            return
        for dest, plan in self._seg_dests.values():
            plan.apply(self._shape_index_scratch, dest)

    def segmentation_info(self) -> dict[str, dict]:
        """Per-output ``idToLabels`` / ``idToSemantics`` info for the requested segmentation outputs."""
        return {name: plan.info for name, (_dest, plan) in self._seg_dests.items()}

    def convert_plane_depth(self):
        """Fill any planar-depth outputs from the ray-hit distance Newton just rendered.

        Newton emits ``distance_to_camera`` (euclidean ray distance). ``depth`` and
        ``distance_to_image_plane`` are the projection of that distance onto the camera's forward
        axis, computed by :meth:`newton.sensors.SensorTiledCamera.Utils.convert_ray_depth_to_forward_depth`.
        No-op when only ``distance_to_camera`` (or no depth output) was requested.
        """
        if self.outputs.depth_image is None:
            return
        for output_name, dest in self._depth_dests.items():
            if output_name in self._PLANE_DEPTH_KINDS:
                self.newton_sensor.utils.convert_ray_depth_to_forward_depth(
                    self.outputs.depth_image,
                    self.camera_transforms,
                    self.camera_rays,
                    out_depth=dest,
                )

    def apply_depth_clipping(self, behavior: str):
        """Apply the renderer's depth-clipping behavior to the depth-family outputs.

        Newton writes ``0.0`` for rays that miss all geometry or fall beyond the far plane
        (``max_distance``), so ``"none"`` and ``"zero"`` both leave that ``0.0`` background. ``"max"``
        replaces the background with the far clip [m] to mirror the RTX renderer's
        :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.depth_clipping_behavior`. No-op when no
        depth output was requested or the camera did not provide a clipping range.
        """
        if behavior != "max" or self.far_clip is None:
            return
        for dest in self._depth_dests.values():
            replace_background_depth_wp(dest, self.far_clip, device=dest.device)

    def update(self, positions: ProxyArray, orientations: ProxyArray, intrinsics: ProxyArray):
        converted_wp = wp.empty_like(orientations)
        convert_camera_frame_orientation_convention_wp(
            src=orientations,
            dst=converted_wp,
            origin="world",
            target="opengl",
            device=self.newton_sensor.model.device,
        )

        self.camera_transforms = wp.empty(
            (1, self.newton_sensor.model.world_count), dtype=wp.transformf, device=self.newton_sensor.model.device
        )
        wp.launch(
            RenderData._update_transforms,
            self.newton_sensor.model.world_count,
            [positions, converted_wp, self.camera_transforms],
            device=self.newton_sensor.model.device,
        )

        if self.camera_rays is None:
            first_focal_length = intrinsics.torch[:, 1, 1][0:1]
            fov_radians_all = 2.0 * torch.atan(self.height / (2.0 * first_focal_length))

            fov_warp = wp.from_torch(fov_radians_all, dtype=wp.float32)
            self.camera_rays = self.newton_sensor.utils.compute_pinhole_camera_rays(self.width, self.height, fov_warp)

    @wp.kernel
    def _update_transforms(
        positions: wp.array(dtype=wp.vec3f),
        orientations: wp.array(dtype=wp.quatf),
        output: wp.array(dtype=wp.transformf, ndim=2),
    ):
        tid = wp.tid()
        output[0, tid] = wp.transformf(positions[tid], orientations[tid])


class NewtonWarpRenderer(BaseRenderer):
    """Newton Warp backend for tiled camera rendering."""

    RenderData = RenderData

    @classmethod
    def provides_temporal_camera_data(cls, data_type: str) -> bool:
        # Pure rasterizer: no temporal accumulation on any output.
        return False

    def __init__(self, cfg: NewtonWarpRendererCfg):
        """Pre-physics initialization."""
        from isaaclab.physics.scene_data_requirements import (
            aggregate_requirements,
            requirement_for_renderer_type,
        )

        self.cfg = cfg
        self.newton_sensor: newton.sensors.SensorTiledCamera | None = None
        # USD stage captured in ``prepare_cameras``; used by the segmentation mapper to read semantics.
        self._stage: Any = None
        # Shared, scene-static segmentation lookup builder, created lazily in ``create_render_data``.
        self._seg_mapper: SegmentationMapper | None = None

        sim = SimulationContext.instance()
        current_req = sim.get_scene_data_requirements()
        renderer_req = requirement_for_renderer_type("newton_warp")
        merged = aggregate_requirements([current_req, renderer_req])
        if merged != current_req:
            sim.update_scene_data_requirements(merged)

    def initialize(self) -> None:
        """Post-physics setup: read the built Newton model and construct the sensor."""
        self._newton_model: newton.Model = NewtonManager.get_model()
        if self._newton_model is None:
            raise RuntimeError(
                "NewtonWarpRenderer requires a Newton model but NewtonManager.get_model() returned None. "
                "This usually means the Newton model failed to build from the USD stage "
                "(e.g., unsupported PhysX schemas such as tendons). "
                "Check the log for earlier Newton model build errors."
            )

        self.newton_sensor = newton.sensors.SensorTiledCamera(
            self._newton_model,
            config=newton.sensors.SensorTiledCamera.RenderConfig(
                enable_textures=self.cfg.enable_textures,
                enable_shadows=self.cfg.enable_shadows,
                enable_ambient_lighting=self.cfg.enable_ambient_lighting,
                enable_backface_culling=self.cfg.enable_backface_culling,
                max_distance=self.cfg.max_distance,
            ),
        )

        # Newton ``v1.2.0rc2`` made shape-BVH construction explicit; ``SensorTiledCamera.update``
        # no longer auto-builds when a non-``None`` state is passed, and the underlying
        # ``RenderContext.render`` raises if ``build_bvh_shape`` was never called for the model.
        # Build it once per model — idempotent across multiple sensors that share ``newton_model``
        # because subsequent calls overwrite the same model-level BVH attributes.
        if self._newton_model.shape_count > 0 and self._newton_model.bvh_shapes is None:
            newton.geometry.build_bvh_shape(self._newton_model, self._newton_model.state())

        if self.cfg.create_default_light:
            self.newton_sensor.utils.create_default_light(enable_shadows=self.cfg.enable_shadows)

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output layout this Newton Warp backend writes.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.supported_output_types`."""

        def seg_spec(colorize: bool) -> RenderBufferSpec:
            # Colorized segmentation is RGBA uint8; raw segmentation is a single int32 id channel.
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
            RenderBufferKind.INSTANCE_SEGMENTATION_FAST: seg_spec(self.cfg.colorize_instance_segmentation),
            RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST: seg_spec(self.cfg.colorize_instance_id_segmentation),
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
        """Create render data for the Newton tiled camera.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`."""
        # Build the shared segmentation mapper once; construction is cheap (per-shape resolution runs
        # lazily when a segmentation output is first requested in ``set_outputs``).
        if self._seg_mapper is None:
            self._seg_mapper = SegmentationMapper(self._newton_model, self._stage, self.cfg)
        render_data = RenderData(self.newton_sensor, spec, seg_mapper=self._seg_mapper, renderer_cfg=self.cfg)
        # Honor the camera's far clipping plane (``spawn.clipping_range[1]``). ``max_distance`` is
        # baked into the sensor at construction from the renderer cfg default; the per-camera far
        # plane is the authoritative value, so push it onto the live render config (documented as
        # mutable for subsequent ``update`` calls). Rays beyond it return no hit and the depth buffer
        # keeps its ``0.0`` background sentinel.
        if render_data.far_clip is not None:
            self.newton_sensor.render_config.max_distance = render_data.far_clip
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

        newton_state: newton.State = NewtonManager.get_state()

        # Refit the shape BVH against the current state since env body poses move every frame.
        # ``build_bvh_shape`` ran once in ``__init__``; ``refit_bvh_shape`` reuses that topology.
        if self.newton_sensor.model.shape_count > 0:
            newton.geometry.refit_bvh_shape(self.newton_sensor.model, newton_state)

        self.newton_sensor.update(
            newton_state,
            render_data.camera_transforms,
            render_data.camera_rays,
            color_image=render_data.outputs.color_image,
            hdr_color_image=render_data.outputs.hdr_color_image,
            albedo_image=render_data.outputs.albedo_image,
            depth_image=render_data.outputs.depth_image,
            normal_image=render_data.outputs.normals_image,
            shape_index_image=render_data.outputs.shape_index_image,
            # ARGB 93% gray to improve visibility of dark objects and align with RTX renderer background
            clear_data=newton.sensors.SensorTiledCamera.ClearData(clear_color=0xFFEEEEEE),
        )

        # Derive planar depth (``depth`` / ``distance_to_image_plane``) from Newton's ray-hit distance.
        render_data.convert_plane_depth()

        # Remap the shape-index buffer into the requested segmentation outputs.
        render_data.convert_segmentation()

        # Apply the configured far-plane clipping behavior to the depth-family outputs.
        render_data.apply_depth_clipping(self.cfg.depth_clipping_behavior)

        # Post-render PPISP: HDR scene-linear → LDR RGBA. Source/destination
        # tensors were bound once in ``set_outputs``.
        if render_data.ppisp_pipeline is not None:
            render_data.ppisp_pipeline.apply(
                render_data._ppisp_hdr_source,
                render_data._ppisp_rgba_dest,
            )

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
        # pixel buffers, mirroring the Isaac RTX renderer's ``camera_data.info`` contract.
        for output_name, info in render_data.segmentation_info().items():
            camera_data.info[output_name] = info

    def cleanup(self, render_data: RenderData | None):
        """Release resources. No-op for Newton Warp.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""
        if render_data:
            render_data.sensor = None
