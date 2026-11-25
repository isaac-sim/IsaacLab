# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton renderer for tiled camera rendering."""

from __future__ import annotations

import dataclasses
import logging
import warnings
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import newton
import warp as wp

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.scene_data import REQUIRES_STAGE_AND_MODEL, SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.utils.version import has_kit
from isaaclab.utils.warp.warp_math import replace_background_depth_wp

from ..physics import NewtonBackendCfg, newton_backend
from .newton_warp_renderer_cfg import NewtonOptixRendererCfg, NewtonWarpRendererCfg
from .segmentation import _NEWTON_NO_HIT_SHAPE_ID, NewtonSegmentationMapper, NewtonSegmentationMapping

if TYPE_CHECKING:
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

logger = logging.getLogger(__name__)

SensorCamera = newton.sensors.SensorCamera


@wp.kernel(enable_backward=False)
def _fill_hdr_background(
    image: wp.array3d(dtype=wp.vec3f),
    shape_ids: wp.array3d(dtype=wp.uint32),
    color: wp.vec3f,
):
    view, row, col = wp.tid()
    if shape_ids[view, row, col] == wp.uint32(wp.static(_NEWTON_NO_HIT_SHAPE_ID)):
        image[view, row, col] = color


@wp.kernel(enable_backward=False)
def _check_shared_intrinsics(intrinsics: wp.array(dtype=wp.mat33f), different: wp.array(dtype=wp.int32)):
    i = wp.tid()
    for row in range(3):
        for column in range(3):
            if wp.abs(intrinsics[i][row, column] - intrinsics[0][row, column]) > 1.0e-4:
                wp.atomic_or(different, 0, 1)


@wp.kernel(enable_backward=False)
def _update_camera_rays(intrinsics: wp.array(dtype=wp.mat33f), rays: wp.array4d(dtype=wp.vec3f)):
    y, x = wp.tid()
    matrix = intrinsics[0]
    direction = wp.vec3f(
        (float(x) + 0.5 - matrix[0, 2]) / matrix[0, 0],
        -(float(y) + 0.5 - matrix[1, 2]) / matrix[1, 1],
        -1.0,
    )
    rays[y, x, 0, 0] = wp.vec3f(0.0)
    rays[y, x, 0, 1] = wp.normalize(direction)


@wp.kernel(enable_backward=False)
def _update_transforms(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    output: wp.array(dtype=wp.transformf),
):
    tid = wp.tid()
    # Convert world camera axes (+X forward, +Z up) to OpenGL (-Z forward, +Y up).
    output[tid] = wp.transformf(positions[tid], orientations[tid] * wp.quatf(0.5, -0.5, -0.5, 0.5))


class RenderData:
    # Back-compat alias for callers of ``RenderData.OutputNames``.
    OutputNames = RenderBufferKind

    # Pixel outputs and their Newton dtypes; Newton writes the camera buffers in place.
    _OUTPUT_MAP: dict[str, tuple[str, type]] = {
        str(RenderBufferKind.RGBA): ("color_image", wp.uint32),
        str(RenderBufferKind.RGB_HDR): ("hdr_color_image", wp.vec3f),
        str(RenderBufferKind.ALBEDO): ("albedo_image", wp.uint32),
        str(RenderBufferKind.NORMALS): ("normals_image", wp.vec3f),
    }

    # Newton's ray depth and planar (forward) depth.
    _RAY_DEPTH_KIND: str = str(RenderBufferKind.DISTANCE_TO_CAMERA)
    _PLANE_DEPTH_KINDS: frozenset[str] = frozenset(
        {str(RenderBufferKind.DEPTH), str(RenderBufferKind.DISTANCE_TO_IMAGE_PLANE)}
    )

    @dataclass
    class CameraOutputs:
        color_image: wp.array(dtype=wp.uint32, ndim=3) = None
        hdr_color_image: wp.array(dtype=wp.vec3f, ndim=3) = None
        albedo_image: wp.array(dtype=wp.uint32, ndim=3) = None
        depth_image: wp.array(dtype=wp.float32, ndim=3) = None
        forward_depth_image: wp.array(dtype=wp.float32, ndim=3) = None
        normals_image: wp.array(dtype=wp.vec3f, ndim=3) = None
        shape_index_image: wp.array(dtype=wp.uint32, ndim=3) = None

    def __init__(
        self,
        model: newton.Model,
        spec: CameraRenderSpec,
        seg_mapper: NewtonSegmentationMapper | None = None,
        renderer_cfg: NewtonWarpRendererCfg | None = None,
    ):
        self._seg_mapper = seg_mapper
        self._renderer_cfg = renderer_cfg
        self.view_count = model.world_count
        self.device = model.device

        self.camera_rays: wp.array(dtype=wp.vec3f, ndim=4) = None
        self.camera_transforms: wp.array(dtype=wp.transformf) = None
        self._intrinsic_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.graph = None
        self.outputs = RenderData.CameraOutputs()
        self._depth_dests: dict[str, wp.array] = {}
        self._seg_dests: dict[str, tuple[wp.array, NewtonSegmentationMapping]] = {}
        self.width = spec.cfg.width
        self.height = spec.cfg.height
        # Newton has no near plane; only the far clip applies.
        spawn = spec.cfg.spawn
        clipping_range = spawn.clipping_range if spawn is not None else None
        self.near_clip: float | None = float(clipping_range[0]) if clipping_range is not None else None
        self.far_clip: float | None = float(clipping_range[1]) if clipping_range is not None else None
        self.render_config: SensorCamera.RenderConfig | None = None

        # Packed RGBA; the default is the RTX renderer's 93% gray background.
        background_color = spec.cfg.background_color
        self.clear_color = 0xFFEEEEEE
        self._hdr_background_color = None
        if background_color is not None:
            clamped = tuple(max(0.0, min(1.0, c)) for c in background_color)
            r, g, b = (round(c * 255) for c in clamped)
            self.clear_color = (0xFF << 24) | (b << 16) | (g << 8) | r
            self._hdr_background_color = wp.vec3f(*newton.utils.color_srgb_to_linear(clamped))

        self._distortion = spawn.distortion if spawn is not None else None

    def _view(self, proxy: ProxyArray, dtype: type) -> wp.array:
        """View an output buffer as an ``(N, H, W)`` array of ``dtype``."""
        array = proxy.warp
        shape = (self.view_count, self.height, self.width)
        return wp.array(ptr=array.ptr, dtype=dtype, shape=shape, device=array.device, copy=False)

    def set_outputs(self, output_data: dict[str, ProxyArray]):
        self._depth_dests = {}
        self._seg_dests = {}
        self.outputs.shape_index_image = None
        self.outputs.depth_image = None
        self.outputs.forward_depth_image = None
        for output_name, proxy in output_data.items():
            if output_name == self._RAY_DEPTH_KIND or output_name in self._PLANE_DEPTH_KINDS:
                dest = self._view(proxy, wp.float32)
                self._depth_dests[output_name] = dest
                if output_name == self._RAY_DEPTH_KIND:
                    self.outputs.depth_image = dest
                elif self.outputs.forward_depth_image is None:
                    self.outputs.forward_depth_image = dest
                continue
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
                self._seg_dests[output_name] = (self._view(proxy, wp.uint32 if colorize else wp.int32), seg_mapping)
                continue
            mapping = self._OUTPUT_MAP.get(output_name)
            if mapping is None:
                if output_name != str(RenderBufferKind.RGB):
                    logger.warning(f"NewtonWarpRenderer - output type {output_name} is not yet supported")
                continue
            field_name, dtype = mapping
            setattr(self.outputs, field_name, self._view(proxy, dtype))
        shape = (self.view_count, self.height, self.width)
        if self._seg_dests:
            self.outputs.shape_index_image = wp.zeros(shape, dtype=wp.uint32, device=self.device)
        # Newton clears HDR to black; a hit mask fills only missed rays with the background.
        if (
            self.outputs.hdr_color_image is not None
            and self._hdr_background_color is not None
            and self.outputs.shape_index_image is None
        ):
            self.outputs.shape_index_image = wp.zeros(shape, dtype=wp.uint32, device=self.device)

    def get_output(self, output_name: str) -> wp.array:
        if output_name in self._depth_dests:
            return self._depth_dests[output_name]
        if output_name in self._seg_dests:
            return self._seg_dests[output_name][0]
        mapping = self._OUTPUT_MAP.get(output_name)
        return getattr(self.outputs, mapping[0]) if mapping is not None else None

    def segmentation_info(self) -> dict[str, dict]:
        """Per-output ``idToLabels`` / ``idToSemantics`` info for the requested segmentation outputs."""
        return {name: seg_mapping.info for name, (_dest, seg_mapping) in self._seg_dests.items()}

    def update(self, positions: ProxyArray, orientations: ProxyArray, intrinsics: ProxyArray):
        # Allocated once, since captured renders read these buffers.
        if self.camera_transforms is None:
            self.camera_transforms = wp.empty(self.view_count, dtype=wp.transformf, device=self.device)
        wp.launch(
            _update_transforms, self.view_count, [positions, orientations, self.camera_transforms], device=self.device
        )

        if self.camera_rays is None:
            if self._distortion is not None:
                self.camera_rays = self._build_distortion_rays()
            else:
                self.camera_rays = wp.empty((self.height, self.width, 1, 2), dtype=wp.vec3f, device=self.device)
                wp.launch(
                    _update_camera_rays,
                    (self.height, self.width),
                    [intrinsics.warp, self.camera_rays],
                    device=self.device,
                )

    def _build_distortion_rays(self) -> wp.array(dtype=wp.vec3f, ndim=4):
        """Return the ``(H, W, 1, 2)`` rays of an OpenCV camera; coefficients are zero without lens distortion."""
        cfg = self._distortion
        coefficient_names = ("k1", "k2", "k3", "k4")
        if cfg.model == "opencvFisheye":
            compute, extra = SensorCamera.compute_camera_rays_fisheye_opencv, {"max_fov": cfg.max_fov}
        else:
            compute, extra = SensorCamera.compute_camera_rays_pinhole_opencv, {}
            coefficient_names += ("k5", "k6", "p1", "p2", "s1", "s2", "s3", "s4")
        coefficients = {
            name: float(getattr(cfg, name)) if cfg.apply_lens_distortion else 0.0 for name in coefficient_names
        }
        return compute(
            self.width,
            self.height,
            float(cfg.fx),
            float(cfg.fy),
            float(cfg.cx),
            float(cfg.cy),
            image_width=float(cfg.image_size[0]),
            image_height=float(cfg.image_size[1]),
            device=self.device,
            **coefficients,
            **extra,
        )


class NewtonWarpRenderer(BaseRenderer):
    """Newton tiled camera renderer, ray traced with Warp or, for :class:`NewtonOptixRendererCfg`, OptiX."""

    RenderData = RenderData

    def __init__(self, cfg: NewtonWarpRendererCfg, *, sim: SimulationContext):
        """Pre-physics initialization."""
        super().__init__(cfg, sim=sim)
        self.newton_sensor: SensorCamera | None = None
        # Shared, scene-static segmentation lookup builder, created lazily in ``create_render_data``.
        self._seg_mapper: NewtonSegmentationMapper | None = None

        self.newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        requires_stage, requires_model = REQUIRES_STAGE_AND_MODEL[cfg.renderer_type]
        sim.requires_usd_stage |= requires_stage
        sim.requires_newton_model |= requires_model

    def initialize(self) -> None:
        """Acquire the clone-built native resource and bind its SDP layout."""
        self.backend = self._sim.get_or_create_backend(self.newton_cfg)
        self._scene_data_provider = self._sim.get_scene_data_provider()
        self._transform_mapping = self._scene_data_provider.create_mapping(list(self.backend.model.body_label))

        self.newton_sensor = SensorCamera(
            self.backend.model,
            backend="optix" if isinstance(self.cfg, NewtonOptixRendererCfg) else "warp",
            default_render_config=SensorCamera.RenderConfig(
                enable_textures=self.cfg.enable_textures,
                enable_shadows=self.cfg.enable_shadows,
                enable_ambient_lighting=self.cfg.enable_ambient_lighting,
                enable_backface_culling=self.cfg.enable_backface_culling,
                max_distance=self.cfg.max_distance,
                render_order=SensorCamera.RenderOrder[self.cfg.render_order.upper()],
                tile_width=self.cfg.tile_rendering_width,
                tile_height=self.cfg.tile_rendering_height,
                block_dim=self.cfg.kernel_block_dim,
            ),
        )
        lighting = getattr(self.backend.model, "isaaclab_lighting", None)
        if lighting is not None and lighting.authored.numpy()[0]:
            self.newton_sensor.set_ambient_light(lighting.ambient.numpy()[0])
            direction = lighting.distant_direction.numpy()[0]
            if any(direction):
                self.newton_sensor.create_default_light(
                    direction=wp.vec3(*direction),
                    color=lighting.distant_color.numpy()[0],
                    enable_shadows=self.cfg.enable_shadows and bool(lighting.shadows.numpy()[0]),
                )
        elif self.cfg.create_default_light:
            self.newton_sensor.create_default_light(enable_shadows=self.cfg.enable_shadows)

    @property
    def visual_material_writer(self):
        """Return the shared Newton model color-writer factory."""
        return self.backend.create_visual_material_writer

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output layout this Newton backend writes.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.supported_output_types`."""
        return self.cfg.supported_output_types()

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """No-op: Newton renders its own model without a stage export.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage`."""
        pass

    def create_render_data(self, spec: CameraRenderSpec) -> RenderData:
        """Create render data for the Newton tiled camera.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`."""
        if spec.cfg.spawn is None or spec.cfg.spawn.distortion is None:
            for path in spec.camera_prim_paths:
                prim = self._sim.stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    continue
                schemas = prim.GetMetadata("apiSchemas")
                if prim.GetAttribute("omni:lensdistortion:model").Get() or (
                    schemas and any("LensDistortion" in name for name in schemas.GetAppliedItems())
                ):
                    warnings.warn(
                        "Newton Warp does not read lens distortion coefficients from camera USD assets. "
                        "Configure spawn.distortion to apply the OpenCV camera model.",
                        UserWarning,
                        stacklevel=2,
                    )
                    break

        segmentation = {
            RenderBufferKind.SEMANTIC_SEGMENTATION: self.cfg.colorize_semantic_segmentation,
            RenderBufferKind.INSTANCE_SEGMENTATION: self.cfg.colorize_instance_segmentation,
        }
        for kind, colorize in segmentation.items():
            if kind in spec.cfg.data_types:
                if self._seg_mapper is None:
                    plan = self._sim.get_clone_plan()
                    self._seg_mapper = NewtonSegmentationMapper(self.backend.model, self._sim.stage, self.cfg, plan)
                self._seg_mapper.build_mapping(kind, bool(colorize))

        render_data = RenderData(self.backend.model, spec, seg_mapper=self._seg_mapper, renderer_cfg=self.cfg)
        far_clip = render_data.far_clip if render_data.far_clip is not None else self.cfg.max_distance
        render_data.render_config = dataclasses.replace(self.newton_sensor.default_render_config, max_distance=far_clip)
        return render_data

    def set_outputs(self, render_data: RenderData, output_data: dict[str, ProxyArray]):
        """Store output buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`."""
        render_data.set_outputs(output_data)

    def update_transforms(self) -> None:
        """No-op: the shared sensor pipeline refreshes transforms immediately before rendering."""
        pass

    def update_geometries(self) -> None:
        """No-op: geometry is read from the Newton state during rendering.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_geometries`."""
        pass

    def update_camera(
        self,
        render_data: RenderData,
        positions: ProxyArray,
        orientations: ProxyArray,
        intrinsics: ProxyArray,
    ):
        """Update camera poses and initialize the ray field on first use.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_camera`."""
        render_data.update(positions, orientations, intrinsics)

    def update_camera_intrinsics(self, render_data: RenderData, intrinsics: wp.array, parameters: wp.array):
        """Regenerate the shared native ray field in place, preserving captured render pointers."""
        if render_data._distortion is not None:
            return  # The camera retains its fixed OpenCV calibration.
        if intrinsics.shape[0] > 1:
            render_data._intrinsic_status.zero_()
            wp.launch(
                _check_shared_intrinsics,
                intrinsics.shape[0],
                [intrinsics, render_data._intrinsic_status],
                device=intrinsics.device,
            )
            if render_data._intrinsic_status.numpy()[0]:
                raise ValueError(
                    "Newton Warp requires identical camera intrinsics across environments: its native ray field "
                    "is shared across worlds. Use a uniform calibration batch or a renderer with per-view intrinsics."
                )
        wp.launch(
            _update_camera_rays,
            (render_data.height, render_data.width),
            [intrinsics, render_data.camera_rays],
            device=intrinsics.device,
        )

    def render(self, render_data: RenderData):
        """Render and write to output buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.render`."""
        backend, provider = self.backend, self._scene_data_provider
        poses = SceneDataFormat.Transform()
        if provider.get_transforms(poses, mapping=self._transform_mapping, count=backend.model.body_count):
            backend.state_0.body_q = poses.transforms
        if backend.geometry_offsets:
            provider.get_geometry_points(output=backend.state_0.particle_q, offsets=backend.geometry_offsets)
        timestamp = provider.backend.transforms_timestamp + provider.backend.geometry_timestamp
        if isinstance(self.cfg, NewtonOptixRendererCfg):
            # OptiX work cannot be recorded in CUDA graphs.
            backend.refit_optix(timestamp)
            self._launch_render(render_data)
        else:
            # Native triangle-mesh updates read back indices and may allocate after pointer swaps.
            use_graph = self.cfg.use_cuda_graph and backend.device.is_cuda and backend.model.tri_count == 0
            backend.refit_bvh(timestamp, use_cuda_graph=use_graph)
            if not use_graph or backend.device.is_capturing:
                self._launch_render(render_data)
            else:
                pointers = backend.state_pointers()
                if render_data.graph is None or render_data.graph[0] != pointers:
                    graph = newton_backend.capture_graph(
                        backend.device, partial(self._launch_render, render_data), relaxed=has_kit()
                    )
                    render_data.graph = pointers, graph
                wp.capture_launch(render_data.graph[1])

    def _launch_render(self, render_data: RenderData) -> None:
        """Render one camera and derive its depth aliases, HDR background and segmentation."""
        outputs = render_data.outputs
        depth_kinds = render_data._depth_dests.keys()
        plane_depth = bool(depth_kinds & render_data._PLANE_DEPTH_KINDS)
        clip_depth = self.cfg.depth_clipping_behavior == "max" and render_data.far_clip is not None
        # Without planar depth, clearing ray depth to the far clip replaces the clipping pass.
        clear_depth = {"clear_depth": render_data.far_clip} if clip_depth and not plane_depth else {}
        self.newton_sensor.update(
            self.backend.state_0,
            render_data.camera_transforms,
            render_data.camera_rays,
            color_image=outputs.color_image,
            hdr_color_image=outputs.hdr_color_image,
            albedo_image=outputs.albedo_image,
            depth_image=outputs.depth_image,
            forward_depth_image=outputs.forward_depth_image,
            normal_image=outputs.normals_image,
            shape_index_image=outputs.shape_index_image,
            clear_data=SensorCamera.ClearData(clear_color=render_data.clear_color, **clear_depth),
            render_config=render_data.render_config,
        )

        if outputs.hdr_color_image is not None and render_data._hdr_background_color is not None:
            wp.launch(
                _fill_hdr_background,
                outputs.hdr_color_image.shape,
                [outputs.hdr_color_image, outputs.shape_index_image, render_data._hdr_background_color],
                device=outputs.hdr_color_image.device,
            )

        if plane_depth:
            for name, dest in render_data._depth_dests.items():
                if name in render_data._PLANE_DEPTH_KINDS and dest.ptr != outputs.forward_depth_image.ptr:
                    wp.copy(dest, outputs.forward_depth_image)
            # "max" replaces Newton's zero background depth with the far clip.
            if clip_depth:
                for dest in render_data._depth_dests.values():
                    replace_background_depth_wp(
                        dest.reshape((*dest.shape, 1)), render_data.far_clip, device=dest.device
                    )

        for dest, seg_mapping in render_data._seg_dests.values():
            seg_mapping.convert_shape_index_to_output(outputs.shape_index_image, dest)

    def read_output(self, render_data: RenderData, camera_data: CameraData) -> None:
        """Publish segmentation metadata; pixel outputs alias the camera buffers bound in :meth:`set_outputs`.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.read_output`."""
        camera_data.info.update(render_data.segmentation_info())

    def cleanup(self, render_data: RenderData | None):
        """Release the camera's buffers and captured query.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""
        if render_data:
            render_data.graph = None

    def close(self) -> None:
        """Release borrowed native handles and SDP bindings after camera cleanup."""
        self.newton_sensor = self.backend = self._scene_data_provider = self._transform_mapping = None
