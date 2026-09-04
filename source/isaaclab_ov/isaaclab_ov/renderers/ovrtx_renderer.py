# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX Renderer implementation."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import sys
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

import numpy as np
import ovstage
import torch
import warp as wp

import isaaclab.utils.warp  # noqa: F401

os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

try:
    from ovrtx import (
        BindingFlag,
        DataAccess,
        Device,
        PrimMode,
        Renderer,
        RendererConfig,
        Semantic,
        TextureStreamingMode,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ovrtx":
        raise
    raise ModuleNotFoundError(
        "The OVRTX renderer requires the optional 'ovrtx' runtime wheel, which is not installed. "
        "Run your command with: uv run --extra ovrtx <command> "
        "(or, manually: python -m pip install 'ovrtx==0.4.1.364340')."
    ) from exc

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp

from isaaclab_ov.stage import create_ovstage

from .ovrtx_annotator_utils import (
    build_instance_id_to_labels_and_semantics,
    build_semantic_id_to_labels,
    decode_semantic_id_map,
    decode_stable_id_map,
    decode_stable_id_semantic_id_map,
)
from .ovrtx_compat import RENDER_VAR_FRAME_KEYS
from .ovrtx_mapping import map_attribute_for_warp_writes
from .ovrtx_renderer_cfg import OVRTXRendererCfg
from .ovrtx_renderer_kernels import (
    compute_cable_points_world_kernel,
    create_camera_transforms_kernel,
    extract_all_tiles_kernel,
    generate_random_colors_from_ids_kernel,
    sync_newton_transforms_kernel,
)
from .ovrtx_shader_cache import redirect_shader_cache
from .ovrtx_usd import build_render_product_as_string, create_scene_partition_attributes, export_stage_to_string
from .visual_materials import OVRTXVisualMaterialWriter

if TYPE_CHECKING:
    from isaaclab_ppisp import PpispPipeline

    from isaaclab.renderers.base_renderer import VisualMaterialBatch

logger = logging.getLogger(__name__)

_LDR_COLOR_VAR = RENDER_VAR_FRAME_KEYS["LdrColor"]
_HDR_COLOR_VAR = RENDER_VAR_FRAME_KEYS["HdrColor"]
_ALBEDO_VAR = RENDER_VAR_FRAME_KEYS["DiffuseAlbedoSD"]
_NORMALS_VAR = RENDER_VAR_FRAME_KEYS["NormalSD"]
_MOTION_VECTORS_VAR = RENDER_VAR_FRAME_KEYS["TargetMotionSD"]
_SEMANTIC_SEGMENTATION_VAR = RENDER_VAR_FRAME_KEYS["SemanticSegmentation"]
_INSTANCE_SEGMENTATION_VAR = RENDER_VAR_FRAME_KEYS["NonStableInstanceSegmentation"]
_SEMANTIC_ID_MAP_VAR = RENDER_VAR_FRAME_KEYS["SemanticIdMap"]
_STABLE_ID_MAP_VAR = RENDER_VAR_FRAME_KEYS["StableIdMap"]
_STABLE_ID_SEMANTIC_ID_MAP_VAR = RENDER_VAR_FRAME_KEYS["StableIdSemanticIdMap"]
_INSTANCE_SEGMENTATION_MAP_VARS = (_STABLE_ID_SEMANTIC_ID_MAP_VAR, _STABLE_ID_MAP_VAR, _SEMANTIC_ID_MAP_VAR)
_DEPTH_VAR_BUFFER_KEYS = {
    RENDER_VAR_FRAME_KEYS["DistanceToImagePlaneSD"]: ("depth", "distance_to_image_plane"),
    RENDER_VAR_FRAME_KEYS["DistanceToCameraSD"]: ("distance_to_camera",),
}
_RTX_MINIMAL_MODES = {
    RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE.value: 1,
    RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL.value: 2,
    RenderBufferKind.SIMPLE_SHADING_FULL_MDL.value: 3,
}
_PPISP_IMPORT_ERROR_MESSAGE = (
    "isaaclab_ppisp is required when CameraCfg.isp_cfg is set. It ships with the Isaac Lab wheel "
    "(`pip install isaaclab`); otherwise install the isaaclab-ppisp extension from the Isaac Lab source checkout."
)
_READ_GPU_TRANSFORMS_ENV = "ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS"
_USE_OVSTAGE_ENV = "ISAAC_LAB_OVRTX_USE_OVSTAGE"
_DISABLE_LINUX_CUDA_CPU_SYNC_ENV = "ISAAC_LAB_OVRTX_DISABLE_LINUX_CUDA_CPU_SYNC"


def ovrtx_use_ovstage_enabled() -> bool:
    value = os.environ.get(_USE_OVSTAGE_ENV, "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(f"Invalid value for environment variable `{_USE_OVSTAGE_ENV}`: {value}. Expected 0 or 1.")
    return value == "1"


def _raise_missing_ppisp_error(exc: ModuleNotFoundError) -> NoReturn:
    if exc.name != "isaaclab_ppisp" and not (exc.name and exc.name.startswith("isaaclab_ppisp.")):
        raise exc
    raise ModuleNotFoundError(_PPISP_IMPORT_ERROR_MESSAGE, name="isaaclab_ppisp") from exc


def _read_gpu_transforms_enabled() -> bool:
    value = os.environ.get(_READ_GPU_TRANSFORMS_ENV, "1").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            f"Invalid value for environment variable `{_READ_GPU_TRANSFORMS_ENV}`: {value}. Expected 0 or 1."
        )
    return value == "1"


def _gpu_side_render_var_sync_enabled() -> bool:
    if not sys.platform.startswith("linux"):
        return True
    value = os.environ.get(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            f"Invalid value for environment variable `{_DISABLE_LINUX_CUDA_CPU_SYNC_ENV}`: {value}. Expected 0 or 1."
        )
    return value == "1"


def _resolve_rtx_minimal_mode(data_types: list[str]) -> int | None:
    filtered = [data_type for data_type in data_types if data_type in _RTX_MINIMAL_MODES]
    if not filtered:
        return None
    if len(filtered) > 1:
        logger.warning("Multiple simple shading data types requested (%s). Using %s.", filtered, filtered[0])
    return _RTX_MINIMAL_MODES[filtered[0]]


def _write_file(output_dir: Path, file_name: str, content: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name
    output_path.write_text(content, encoding="utf-8")
    logger.info("Wrote USD file: %s", output_path)


class OVRTXRenderData:
    def __init__(self, spec: CameraRenderSpec, device):
        self.width = spec.cfg.width
        self.height = spec.cfg.height
        self.num_envs = spec.num_instances
        self.data_types = spec.cfg.data_types or ["rgb"]
        self.num_cols = math.ceil(math.sqrt(self.num_envs))
        self.num_rows = math.ceil(self.num_envs / self.num_cols)
        self.warp_buffers: dict[str, wp.array] = {}
        self.renderer_info: dict[str, Any] = {}
        self.ppisp_pipeline: PpispPipeline | None = None
        if spec.cfg.isp_cfg is not None:
            try:
                from isaaclab_ppisp import PpispPipeline
            except ModuleNotFoundError as exc:
                _raise_missing_ppisp_error(exc)
            self.ppisp_pipeline = PpispPipeline(spec.cfg.isp_cfg)


class OVRTXRenderer(BaseRenderer):
    cfg: OVRTXRendererCfg

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        instance = (
            RenderBufferSpec(4, wp.uint8) if self.cfg.colorize_instance_segmentation else RenderBufferSpec(1, wp.int32)
        )
        semantic = (
            RenderBufferSpec(4, wp.uint8) if self.cfg.colorize_semantic_segmentation else RenderBufferSpec(1, wp.int32)
        )
        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.ALBEDO: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_FULL_MDL: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SEMANTIC_SEGMENTATION: semantic,
            RenderBufferKind.INSTANCE_SEGMENTATION: instance,
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.MOTION_VECTORS: RenderBufferSpec(2, wp.float32),
        }

    def __init__(self, cfg: OVRTXRendererCfg):
        self.cfg = cfg
        self._device = "cuda:0"
        self._warp_device = None
        self._render_product_paths = []
        self._object_newton_indices = self._object_scales = None
        self._object_scales_by_path = {}
        self._deformable_particle_offsets = []
        self._deformable_particle_counts = []
        self._particle_visual_offsets = []
        self._particle_visual_counts = []
        self._cable_segment_counts = []
        self._cable_max_points = 0
        self._cable_shape_ids = self._cable_offsets = self._cable_counts = self._cable_points = None
        self._initialized_scene = False
        self._exported_usd_string = self._camera_rel_path = None
        self._output_id_color_buffers = {}
        self._clone_plan = None
        self._visual_material_writer_ref = None
        self._use_ovstage = ovrtx_use_ovstage_enabled()
        self._init_fields()
        config = RendererConfig(
            log_file_path=cfg.log_file_path,
            log_level=cfg.log_level,
            read_gpu_transforms=_read_gpu_transforms_enabled(),
            keep_system_alive=True,
            suppress_deprecation_warnings=True,
            texture_streaming_mode=TextureStreamingMode.SYNCHRONOUS,
        )
        redirect_shader_cache(config)
        self._renderer = Renderer(config)
        if not self._renderer:
            raise RuntimeError("Failed to create OVRTX Renderer")

    def _create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]):
        if not self._initialized_scene:
            raise RuntimeError("OVRTX must ingest its detached scene before material writes are compiled.")
        writer = OVRTXVisualMaterialWriter(self, batches)
        self._visual_material_writer_ref = weakref.ref(writer)
        return writer

    @property
    def visual_material_writer(self):
        return self._create_visual_material_writer

    def prepare_cameras(self, stage: Any, spec: CameraRenderSpec) -> None:
        if spec.cfg.isp_cfg is None:
            return
        try:
            from isaaclab_ppisp import apply_rtx_exposure_overrides, resolve_and_normalize
        except ModuleNotFoundError as exc:
            _raise_missing_ppisp_error(exc)
        path = spec.camera_prim_paths[0] if spec.camera_prim_paths else None
        spec.cfg.isp_cfg = resolve_and_normalize(spec.cfg.isp_cfg, stage, path)
        if spec.cfg.isp_cfg is not None and spec.camera_prim_paths:
            apply_rtx_exposure_overrides(stage, list(spec.camera_prim_paths))

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        if stage is None:
            return
        self._clone_plan = SimulationContext.instance().get_clone_plan()
        if self._clone_plan is None or self._clone_plan.env_ids is None or self._clone_plan.positions is None:
            raise RuntimeError("Clone plan with environment ids and positions is required when preparing OVRTX stage")
        if not torch.equal(self._clone_plan.env_ids, torch.arange(num_envs, device=self._clone_plan.env_ids.device)):
            raise RuntimeError("OVRTX requires ClonePlan environment ids ordered from zero.")
        if self.cfg.temp_usd_dir is not None:
            _write_file(Path(self.cfg.temp_usd_dir), "pre_ovrtx_renderer_stage.usda", stage.ExportToString())
        create_scene_partition_attributes(stage, num_envs)
        self._capture_object_scales(stage)
        self._exported_usd_string = export_stage_to_string(
            stage, num_envs, source_paths=self._clone_plan.sources, keep_env_roots=not self._use_ovstage
        )

    def _capture_object_scales(self, stage):
        self._object_scales_by_path.clear()
        from pxr import Gf, Usd, UsdGeom

        root = stage.GetPrimAtPath("/World/envs")
        if not root.IsValid():
            return
        cache = UsdGeom.XformCache()
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Xformable):
                value = Gf.Transform(cache.GetLocalToWorldTransform(prim)).GetScale()
                scale = tuple(float(value[index]) for index in range(3))
                if not all(math.isclose(axis, 1.0, rel_tol=1e-6, abs_tol=1e-6) for axis in scale):
                    self._object_scales_by_path[str(prim.GetPath())] = scale

    def _create_object_scale_array(self, paths):
        return wp.array(
            [self._object_scales_by_path.get(path, (1.0, 1.0, 1.0)) for path in paths],
            dtype=wp.vec3f,
            device=self._device,
        )

    def _init_fields_legacy(self):
        self._camera_xform_binding = self._object_xform_binding = None
        self._deformable_points_binding = self._particle_points_binding = self._cable_points_binding = None
        self._particle_workaround_applied = False
        self._cable_point_slices = []

    def _initialize_from_spec_legacy(self, spec):
        data_types = spec.cfg.data_types or ["rgb"]
        if spec.cfg.isp_cfg is not None and "rgb_hdr" not in data_types:
            data_types = [*data_types, "rgb_hdr"]
        self._camera_rel_path = spec.camera_path_relative_to_env_0
        product, path = build_render_product_as_string(
            width=spec.cfg.width,
            height=spec.cfg.height,
            num_envs=spec.num_instances,
            data_types=data_types,
            minimal_mode=_resolve_rtx_minimal_mode(data_types),
            camera_rel_path=self._camera_rel_path,
            background_color=getattr(spec.cfg, "background_color", None),
            device_id=self._warp_device.ordinal,
            enable_shadows=self.cfg.enable_shadows,
        )
        combined = self._exported_usd_string + "\n\n" + product
        self._exported_usd_string = None
        self._render_product_paths.append(path)
        self._renderer.open_usd_from_string(combined)
        cameras = [f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(spec.num_instances)]
        if spec.num_instances > 1:
            self._clone_sources_in_ovrtx()
            self._update_scene_partitions_after_clone(spec.num_instances)
            self._renderer.write_array_attribute(prim_paths=[path], attribute_name="camera", tensors=[cameras])
        self._initialized_scene = True
        self._camera_xform_binding = self._renderer.bind_attribute(
            prim_paths=cameras,
            attribute_name="omni:xform",
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.EXISTING_ONLY,
        )
        self._renderer.write_attribute(
            prim_paths=cameras,
            attribute_name="omni:resetXformStack",
            tensor=np.full(spec.num_instances, True, dtype=np.bool_),
        )
        self._setup_xform_bindings()
        self._setup_deformable_bindings(spec.num_instances)
        self._setup_particle_bindings()
        self._setup_cable_bindings()

    def _clone_sources_in_ovrtx(self):
        plan = self._clone_plan
        ids = plan.env_ids.detach().cpu()
        mask = plan.clone_mask.detach().cpu()
        for row, (source, destination) in enumerate(zip(plan.sources, plan.destinations, strict=True)):
            targets = [
                destination.format(int(i)) for i in ids[mask[row]].tolist() if destination.format(int(i)) != source
            ]
            if targets:
                self._renderer.clone_usd(source, targets)
        paths = [f"/World/envs/env_{i}" for i in ids.tolist()]
        xforms = np.tile(np.eye(4, dtype=np.float64), (len(ids), 1, 1))
        xforms[:, 3, :3] = plan.positions.cpu().numpy()
        self._renderer.write_attribute(
            prim_paths=paths,
            attribute_name="omni:xform",
            tensor=xforms,
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.MUST_EXIST,
        )

    def _update_scene_partitions_after_clone(self, count):
        tokens = [f"env_{i}" for i in range(count)]
        self._renderer.write_attribute(
            [f"/World/envs/env_{i}" for i in range(count)],
            "primvars:omni:scenePartition",
            tokens,
            semantic=Semantic.TOKEN_STRING,
        )
        self._renderer.write_attribute(
            [f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(count)],
            "omni:scenePartition",
            tokens,
            semantic=Semantic.TOKEN_STRING,
        )

    def _setup_xform_bindings_legacy(self):
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return
        model = NewtonManager.get_model()
        if model is None or getattr(model, "body_label", None) is None:
            return
        pairs = [
            (i, p)
            for i, p in enumerate(model.body_label)
            if "/World/envs/" in p and self._camera_rel_path not in p and "GroundPlane" not in p
        ]
        if not pairs:
            return
        indices, paths = zip(*pairs, strict=True)
        self._object_xform_binding = self._renderer.bind_attribute(
            prim_paths=list(paths),
            attribute_name="omni:xform",
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.EXISTING_ONLY,
        )
        self._renderer.write_attribute(
            prim_paths=list(paths),
            attribute_name="omni:resetXformStack",
            tensor=np.full(len(paths), True, dtype=np.bool_),
        )
        self._object_newton_indices = wp.array(indices, dtype=wp.int32, device=self._device)
        self._object_scales = self._create_object_scale_array(list(paths))

    def _setup_deformable_bindings_legacy(self, num_envs):
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return
        paths = []
        for entry in NewtonManager._deformable_registry:
            if len(entry.particle_offsets) != num_envs:
                raise RuntimeError(f"OVRTX expects one particle offset per environment ({num_envs})")
            for i, offset in enumerate(entry.particle_offsets):
                self._deformable_particle_offsets.append(offset)
                self._deformable_particle_counts.append(entry.particles_per_body)
                paths.append(re.sub(r"(?<=[Ee]nv_)(?:\[\^/\][*+]|\.\*)", str(i), entry.vis_mesh_prim_path))
        if paths:
            self._deformable_points_binding = self._setup_array_binding(paths)

    def _setup_particle_bindings_legacy(self):
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return
        records = NewtonManager._particle_visual_prims
        self._particle_visual_offsets = [r.offset for r in records.values()]
        self._particle_visual_counts = [r.count for r in records.values()]
        if records:
            self._particle_points_binding = self._setup_array_binding(list(records))

    def _setup_array_binding(self, paths):
        count = len(paths)
        self._renderer.write_attribute(
            prim_paths=paths,
            attribute_name="omni:resetXformStack",
            tensor=np.full(count, True, dtype=np.bool_),
            prim_mode=PrimMode.MUST_EXIST,
        )
        self._renderer.write_attribute(
            prim_paths=paths,
            attribute_name="omni:xform",
            tensor=np.tile(np.eye(4), (count, 1, 1)),
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.MUST_EXIST,
        )
        return self._renderer.bind_array_attribute(
            prim_paths=paths,
            attribute_name="points",
            dtype=np.float32,
            shape=(3,),
            prim_mode=PrimMode.MUST_EXIST,
            flags=BindingFlag.OPTIMIZE,
        )

    def _setup_cable_bindings_legacy(self):
        found = self._discover_cable_segment_bindings()
        if found is None:
            return
        paths, ids, offsets, counts = found
        self._cable_points_binding = self._setup_array_binding(paths)
        self._allocate_cable_device_buffers(ids, offsets, counts)
        self._cable_point_slices = [
            self._cable_points[o + i : o + i + c + 1] for i, (o, c) in enumerate(zip(offsets, counts, strict=True))
        ]

    def create_render_data(self, spec):
        self._warp_device = wp.get_device(spec.device)
        self._device = str(self._warp_device)
        if not self._initialized_scene:
            self._initialize_from_spec(spec)
        return OVRTXRenderData(spec, self._device)

    def set_outputs(self, data, outputs):
        data.warp_buffers = {name: proxy.warp for name, proxy in outputs.items() if name != str(RenderBufferKind.RGB)}
        if data.ppisp_pipeline is not None and "rgb_hdr" not in data.warp_buffers:
            ref = next(iter(outputs.values()))
            data.warp_buffers["rgb_hdr"] = wp.zeros(
                (data.num_envs, data.height, data.width, 3), dtype=wp.float32, device=ref.device
            )
        if data.ppisp_pipeline is not None and "rgba" not in data.warp_buffers:
            raise ValueError("OVRTX renderer ISP requires an LDR rgba output destination")

    def _update_transforms_legacy(self):
        if self._object_xform_binding is None:
            return
        from isaaclab_newton.physics import NewtonManager

        state = NewtonManager.get_state()
        if state is None:
            raise RuntimeError("Newton state should not be None")
        with map_attribute_for_warp_writes(self._object_xform_binding, self._warp_device, wp.mat44d) as transforms:
            wp.launch(
                sync_newton_transforms_kernel,
                len(self._object_newton_indices),
                inputs=[transforms, self._object_newton_indices, state.body_q, self._object_scales],
                device=self._device,
            )

    def _update_geometries_legacy(self):
        if self._deformable_points_binding:
            self._write_particle_q_slices(
                self._deformable_points_binding, self._deformable_particle_offsets, self._deformable_particle_counts
            )
        if self._particle_points_binding:
            self._write_particle_q_slices(
                self._particle_points_binding, self._particle_visual_offsets, self._particle_visual_counts
            )
        if self._cable_points_binding:
            self._compute_cable_points_world()
            self._cable_points_binding.write(
                cast(Any, self._cable_point_slices),
                data_access=DataAccess.ASYNC,
                cuda_stream=self._warp_device.stream.cuda_stream,
            )

    def _write_particle_q_slices(self, binding, offsets, counts):
        from isaaclab_newton.physics import NewtonManager

        state = NewtonManager.get_state()
        slices = [state.particle_q[o : o + c] for o, c in zip(offsets, counts, strict=True)]
        binding.write(cast(Any, slices), data_access=DataAccess.ASYNC, cuda_stream=self._warp_device.stream.cuda_stream)

    def _update_camera_legacy(self, render_data, positions, orientations, intrinsics):
        converted = wp.empty(len(positions), dtype=wp.quatf, device=self._device)
        convert_camera_frame_orientation_convention_wp(
            src=orientations.warp, dst=converted, origin="world", target="opengl", device=self._device
        )
        transforms = wp.zeros(len(positions), dtype=wp.mat44d, device=self._device)
        wp.launch(
            create_camera_transforms_kernel,
            len(positions),
            inputs=[positions, converted, transforms],
            device=self._device,
        )
        if self._camera_xform_binding:
            with map_attribute_for_warp_writes(self._camera_xform_binding, self._warp_device, wp.mat44d) as view:
                wp.copy(view, transforms)

    def read_output(self, render_data, camera_data):
        for name in camera_data.info:
            camera_data.info[name] = render_data.renderer_info.get(name)

    @contextlib.contextmanager
    def _map_render_var_to_dlpack(self, render_var) -> Iterator[wp.array]:
        gpu = _gpu_side_render_var_sync_enabled()
        with render_var.map(
            device=Device.CUDA, sync_stream=self._warp_device.stream.cuda_stream if gpu else 0
        ) as mapping:
            if not gpu:
                mapping.wait()
            yield wp.from_dlpack(mapping)

    def _launch_extract_all_tiles(self, data, tiled, output):
        if output.shape[-1] > tiled.shape[-1]:
            raise ValueError("Output buffer has more channels than tiled buffer")
        wp.launch(
            extract_all_tiles_kernel,
            (data.num_envs, data.height, data.width),
            inputs=[tiled, output, data.num_cols, data.width, data.height],
            device=self._device,
        )

    def _extract_rgba_tiles(self, data, tiled, outputs, key, suffix=""):
        self._launch_extract_all_tiles(data, tiled, outputs[key])

    def _process_id_segmentation_render_var(self, data, frame, outputs, render_var_key, key, colorize):
        var = frame.render_vars.get(render_var_key)
        if var is None or key not in outputs:
            return
        with self._map_render_var_to_dlpack(var) as tiled:
            if colorize:
                colors = self._output_id_color_buffers.get(key)
                if colors is None or colors.shape != tiled.shape:
                    colors = wp.zeros(tiled.shape, dtype=wp.uint32, device=self._device)
                wp.launch(
                    generate_random_colors_from_ids_kernel, tiled.shape, inputs=[tiled, colors], device=self._device
                )
                self._output_id_color_buffers[key] = colors
                tensor = wp.to_torch(colors)
                view = tensor.view(torch.uint8)
                if tensor.dim() == 2:
                    view = view.reshape(*tensor.shape, 4)
                tiled = wp.from_torch(view, dtype=wp.uint8)
            elif tiled.ndim == 2:
                tiled = tiled.reshape((*tiled.shape, 1))
            self._launch_extract_all_tiles(data, tiled, outputs[key])

    def _process_render_frame(self, data, frame, outputs):
        data.renderer_info.clear()
        for key, output in (
            (_LDR_COLOR_VAR, "rgba"),
            (_ALBEDO_VAR, "albedo"),
            (_NORMALS_VAR, "normals"),
            (_MOTION_VECTORS_VAR, "motion_vectors"),
        ):
            var = frame.render_vars.get(key)
            if var is not None and output in outputs:
                with self._map_render_var_to_dlpack(var) as tiled:
                    self._launch_extract_all_tiles(data, tiled, outputs[output])
        for key, names in _DEPTH_VAR_BUFFER_KEYS.items():
            var = frame.render_vars.get(key)
            if var is not None:
                with self._map_render_var_to_dlpack(var) as tiled:
                    if tiled.dtype == wp.uint32:
                        tiled = wp.from_torch(wp.to_torch(tiled).view(torch.float32), dtype=wp.float32)
                    for name in names:
                        if name in outputs:
                            self._launch_extract_all_tiles(data, tiled, outputs[name])
        hdr = frame.render_vars.get(_HDR_COLOR_VAR)
        if hdr is not None and "rgb_hdr" in outputs:
            with self._map_render_var_to_dlpack(hdr) as tiled:
                self._launch_extract_all_tiles(data, tiled, outputs["rgb_hdr"])
        self._process_id_segmentation_render_var(
            data,
            frame,
            outputs,
            _SEMANTIC_SEGMENTATION_VAR,
            "semantic_segmentation",
            self.cfg.colorize_semantic_segmentation,
        )
        self._process_id_segmentation_render_var(
            data,
            frame,
            outputs,
            _INSTANCE_SEGMENTATION_VAR,
            "instance_segmentation",
            self.cfg.colorize_instance_segmentation,
        )
        semantic_map = frame.render_vars.get(_SEMANTIC_ID_MAP_VAR)
        if "semantic_segmentation" in outputs and semantic_map is not None:
            with semantic_map.map(device=Device.CPU) as mapping:
                labels = decode_semantic_id_map(np.from_dlpack(mapping))
            data.renderer_info["semantic_segmentation"] = {
                "idToLabels": build_semantic_id_to_labels(
                    labels, colorize=self.cfg.colorize_semantic_segmentation, device=self._device
                )
            }
        if "instance_segmentation" in outputs:
            resolved = {key: frame.render_vars.get(key) for key in _INSTANCE_SEGMENTATION_MAP_VARS}
            missing = [key for key, value in resolved.items() if value is None]
            if missing:
                raise RuntimeError(f"instance_segmentation render vars missing: {missing}")
            with resolved[_STABLE_ID_SEMANTIC_ID_MAP_VAR].map(device=Device.CPU) as mapping:
                stable_sem = decode_stable_id_semantic_id_map(np.from_dlpack(mapping))
            with resolved[_STABLE_ID_MAP_VAR].map(device=Device.CPU) as mapping:
                paths = decode_stable_id_map(np.from_dlpack(mapping))
            with resolved[_SEMANTIC_ID_MAP_VAR].map(device=Device.CPU) as mapping:
                labels = decode_semantic_id_map(np.from_dlpack(mapping))
            ids, semantics = build_instance_id_to_labels_and_semantics(
                stable_sem, paths, labels, colorize=self.cfg.colorize_instance_segmentation, device=self._device
            )
            data.renderer_info["instance_segmentation"] = {"idToLabels": ids, "idToSemantics": semantics}

    def _finish_render(self, data, products):
        path = self._render_product_paths[0]
        if path in products and products[path].frames:
            self._process_render_frame(data, products[path].frames[0], data.warp_buffers)
        if data.ppisp_pipeline is not None:
            data.ppisp_pipeline.apply(data.warp_buffers["rgb_hdr"], data.warp_buffers["rgba"])

    def _render_legacy(self, data):
        writer = self._visual_material_writer_ref() if self._visual_material_writer_ref else None
        try:
            if writer:
                writer.publish()
            products = self._renderer.step(render_products=set(self._render_product_paths), delta_time=1 / 60)
        finally:
            if writer:
                writer.drain()
        self._finish_render(data, products)

    def _close_legacy(self):
        for name in (
            "_camera_xform_binding",
            "_object_xform_binding",
            "_deformable_points_binding",
            "_particle_points_binding",
            "_cable_points_binding",
        ):
            binding = getattr(self, name)
            if binding:
                with contextlib.suppress(Exception):
                    binding.unbind()
            setattr(self, name, None)
        if self._renderer:
            with contextlib.suppress(Exception):
                self._renderer.reset_stage()
        self._renderer = None
        self._initialized_scene = False

    def _init_fields(self):
        self._init_fields_ovstage() if self._use_ovstage else self._init_fields_legacy()

    def _initialize_from_spec(self, spec):
        self._initialize_from_spec_ovstage(spec) if self._use_ovstage else self._initialize_from_spec_legacy(spec)

    def _setup_xform_bindings(self):
        self._setup_xform_bindings_ovstage() if self._use_ovstage else self._setup_xform_bindings_legacy()

    def _setup_deformable_bindings(self, count):
        self._setup_deformable_bindings_ovstage(count) if self._use_ovstage else self._setup_deformable_bindings_legacy(
            count
        )

    def _setup_particle_bindings(self):
        self._setup_particle_bindings_ovstage() if self._use_ovstage else self._setup_particle_bindings_legacy()

    def _setup_cable_bindings(self):
        self._setup_cable_bindings_ovstage() if self._use_ovstage else self._setup_cable_bindings_legacy()

    @staticmethod
    def _discover_cable_segment_bindings():
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return None
        records = NewtonManager.collect_cable_segment_shape_ids()
        if not records:
            return None
        paths, ids, offsets, counts = [], [], [], []
        for path, shape_ids in records.items():
            paths.append(path)
            offsets.append(len(ids))
            counts.append(len(shape_ids))
            ids.extend(shape_ids)
        return paths, ids, offsets, counts

    def _allocate_cable_device_buffers(self, ids, offsets, counts):
        self._cable_shape_ids = wp.array(ids, dtype=wp.int32, device=self._device)
        self._cable_offsets = wp.array(offsets, dtype=wp.int32, device=self._device)
        self._cable_counts = wp.array(counts, dtype=wp.int32, device=self._device)
        self._cable_segment_counts = counts
        self._cable_max_points = max(counts) + 1
        self._cable_points = wp.zeros(sum(c + 1 for c in counts), dtype=wp.vec3f, device=self._device)

    def _compute_cable_points_world(self):
        from isaaclab_newton.physics import NewtonManager

        model, state = NewtonManager.get_model(), NewtonManager.get_state()
        wp.launch(
            compute_cable_points_world_kernel,
            (len(self._cable_segment_counts), self._cable_max_points),
            inputs=[
                self._cable_shape_ids,
                self._cable_offsets,
                self._cable_counts,
                model.shape_body,
                state.body_q,
                model.shape_transform,
                model.shape_scale,
                self._cable_points,
            ],
            device=self._device,
        )

    def update_transforms(self):
        self._update_transforms_ovstage() if self._use_ovstage else self._update_transforms_legacy()

    def update_geometries(self):
        self._update_geometries_ovstage() if self._use_ovstage else self._update_geometries_legacy()

    def update_camera(self, data, positions, orientations, intrinsics):
        (self._update_camera_ovstage if self._use_ovstage else self._update_camera_legacy)(
            data, positions, orientations, intrinsics
        )

    def render(self, data):
        (self._render_ovstage if self._use_ovstage else self._render_legacy)(data)

    def cleanup(self, data):
        if data:
            data.warp_buffers.clear()
            data.renderer_info.clear()
            data.ppisp_pipeline = None

    def close(self):
        (self._close_ovstage if self._use_ovstage else self._close_legacy)()
        self._visual_material_writer_ref = None

    def _init_fields_ovstage(self):
        self._stage = self._stage_paths = self._ovstage_exit_stack = None
        self._current_ordinal = 0
        self._camera_xform_query = self._camera_paths_list = self._object_xform_query = self._object_paths_list = None
        self._deformable_points_query = self._deformable_paths_list = self._particle_points_query = (
            self._particle_paths_list
        ) = None
        self._cable_points_query = self._cable_paths_list = None
        self._cable_point_tensors = []

    def _initialize_from_spec_ovstage(self, spec):
        data_types = spec.cfg.data_types or ["rgb"]
        if spec.cfg.isp_cfg is not None and "rgb_hdr" not in data_types:
            data_types.append("rgb_hdr")
        self._camera_rel_path = spec.camera_path_relative_to_env_0
        product, path = build_render_product_as_string(
            width=spec.cfg.width,
            height=spec.cfg.height,
            num_envs=spec.num_instances,
            data_types=data_types,
            minimal_mode=_resolve_rtx_minimal_mode(data_types),
            camera_rel_path=self._camera_rel_path,
            device_id=self._warp_device.ordinal,
            enable_shadows=self.cfg.enable_shadows,
        )
        combined = self._exported_usd_string + "\n\n" + product
        self._exported_usd_string = None
        self._render_product_paths.append(path)
        self._ovstage_exit_stack = contextlib.ExitStack()
        self._stage = self._ovstage_exit_stack.enter_context(create_ovstage("isaaclab.ovrtx"))
        self._stage_paths = self._ovstage_exit_stack.enter_context(ovstage.PathDictionary(self._stage))
        self._current_ordinal = 1
        ovstage.population.open_usd_from_string(
            self._stage, combined, ordinal=1, domains=ovstage.PopulationDomain.RENDERING
        )
        if spec.num_instances > 1:
            self._clone_sources_ovstage()
            self._update_scene_partitions_after_clone_ovstage(spec.num_instances)
        cameras = [f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(spec.num_instances)]
        paths = self._stage_paths.create_path_list_from_strings(cameras)
        self._camera_paths_list = paths
        self._camera_xform_query = self._stage.query_from_path_list(paths)
        self._setup_xform_bindings_ovstage()
        self._setup_deformable_bindings_ovstage(spec.num_instances)
        self._setup_particle_bindings_ovstage()
        self._setup_cable_bindings_ovstage()
        self._stage.advance_write_floor(ordinal=1).wait()
        self._renderer.attach_ovstage(self._stage)
        self._current_ordinal = 2
        self._initialized_scene = True

    def _clone_sources_ovstage(self):
        plan = self._clone_plan
        for row, (source, destination) in enumerate(zip(plan.sources, plan.destinations, strict=True)):
            ids = plan.env_ids[plan.clone_mask[row]].tolist()
            targets = [destination.format(int(i)) for i in ids if destination.format(int(i)) != source]
            if targets:
                self._stage.clone(source, targets, ordinal=self._current_ordinal)

    def _update_scene_partitions_after_clone_ovstage(self, count):
        pass

    def _setup_xform_bindings_ovstage(self):
        pass

    def _setup_deformable_bindings_ovstage(self, count):
        pass

    def _setup_particle_bindings_ovstage(self):
        pass

    def _setup_cable_bindings_ovstage(self):
        pass

    def _update_transforms_ovstage(self):
        pass

    def _update_geometries_ovstage(self):
        pass

    def _update_camera_ovstage(self, data, positions, orientations, intrinsics):
        self._update_camera_legacy(data, positions, orientations, intrinsics)

    def _render_ovstage(self, data):
        self._stage.advance_write_floor(ordinal=self._current_ordinal).wait()
        products = self._renderer.step(
            render_products=set(self._render_product_paths), delta_time=1 / 60, ordinal=self._current_ordinal
        )
        self._current_ordinal += 1
        self._finish_render(data, products)

    def _close_ovstage(self):
        if self._renderer is not None and self._stage is not None:
            self._renderer.detach_ovstage()
        self._renderer = None
        if self._ovstage_exit_stack:
            self._ovstage_exit_stack.close()
        self._stage = self._stage_paths = self._ovstage_exit_stack = None
        self._initialized_scene = False
