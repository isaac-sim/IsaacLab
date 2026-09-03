# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX Renderer implementation.

How it fits together
--------------------
- **ovrtx_renderer.py** (this file): Orchestrates the pipeline. Owns the OVRTX Renderer,
  USD loading/cloning, camera and object bindings, and output buffers. Each frame it:
  updates camera/object transforms (using kernels), steps the renderer, then extracts
  tiles from the tiled framebuffer (kernels).

- **ovrtx_renderer_kernels.py**: Warp GPU kernels for OVRTX rendering pipeline.

- **ovrtx_usd.py**: USD helpers for OVRTX: render var config, camera injection, etc.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import sys
import weakref
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

logger = logging.getLogger(__name__)

import numpy as np
import ovstage
import torch
import warp as wp

import isaaclab.utils.warp  # noqa: F401  # initializes Warp runtime

# The ovrtx C library links to its own version of the USD libraries. Having
# the pxr Python package available can cause the C library to load an
# incompatible version of libusd, potentially leading to undefined behavior.
# By setting OVRTX_SKIP_USD_CHECK, we prevent the C library from loading the pxr Python package.
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

from isaaclab.cloner import ClonePlan
from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp

from isaaclab_ov.stage import (
    create_ovstage,
    points_tensor_from_warp,
    xform_tensor_from_numpy,
    xform_tensor_from_warp,
)

from .ovrtx_annotator_utils import (
    build_instance_id_to_labels_and_semantics,
    build_semantic_id_to_labels,
    decode_semantic_id_map,
    decode_stable_id_map,
    decode_stable_id_semantic_id_map,
)
from .ovrtx_mapping import cuda_device_id, map_attribute_for_warp_writes
from .ovrtx_renderer_cfg import OVRTXRendererCfg
from .ovrtx_renderer_kernels import (
    compute_cable_points_world_kernel,
    create_camera_transforms_kernel,
    extract_all_tiles_kernel,
    generate_random_colors_from_ids_kernel,
    sync_newton_transforms_kernel,
)
from .ovrtx_usd import (
    build_render_product_as_string,
    create_scene_partition_attributes,
    export_stage_to_string,
)
from .visual_materials import OVRTXVisualMaterialWriter

if TYPE_CHECKING:
    from isaaclab_ppisp import PpispPipeline

    from isaaclab.renderers.base_renderer import VisualMaterialBatch
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

from isaaclab.renderers.camera_render_spec import CameraRenderSpec

# Maps depth render-var sources to compatible output buffers.
_DEPTH_VAR_BUFFER_KEYS: dict[str, tuple[str, ...]] = {
    "DistanceToImagePlaneSD": ("depth", "distance_to_image_plane"),
    "DistanceToCameraSD": ("distance_to_camera",),
}

# The resolved integer value is assigned to the ``omni:rtx:minimal:mode`` attribute of the render product.
_RTX_MINIMAL_MODES = {
    RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE.value: 1,
    RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL.value: 2,
    RenderBufferKind.SIMPLE_SHADING_FULL_MDL.value: 3,
}

_PPISP_IMPORT_ERROR_MESSAGE = (
    "isaaclab_ppisp is required when CameraCfg.isp_cfg is set. "
    "It ships with the Isaac Lab wheel (`pip install isaaclab`); otherwise install the "
    "isaaclab-ppisp extension from the Isaac Lab source checkout."
)
_READ_GPU_TRANSFORMS_ENV = "ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS"


# Runtime environment variable used to enable the ovstage code path for ovrtx.
_USE_OVSTAGE_ENV = "ISAAC_LAB_OVRTX_USE_OVSTAGE"


# Opts Linux out of the host wait, onto the same GPU-side ordering every other platform uses.
# See :meth:`OVRTXRenderer._map_render_var_to_dlpack`.
_DISABLE_LINUX_CUDA_CPU_SYNC_ENV = "ISAAC_LAB_OVRTX_DISABLE_LINUX_CUDA_CPU_SYNC"


def ovrtx_use_ovstage_enabled() -> bool:
    """Return whether the ovstage scene-ownership path should be used.

    Enabled by ``ISAAC_LAB_OVRTX_USE_OVSTAGE=1``. Defaults to ``0`` so existing deployments are
    unaffected until ovstage is explicitly opted into.

    Raises:
        ValueError: If the environment variable is set to anything other than ``0`` or ``1``.
    """
    value = os.environ.get(_USE_OVSTAGE_ENV, "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(f"Invalid value for environment variable `{_USE_OVSTAGE_ENV}`: {value}. Expected 0 or 1.")
    return value == "1"


def _raise_missing_ppisp_error(exc: ModuleNotFoundError) -> NoReturn:
    # Only translate missing isaaclab_ppisp imports into the optional-dependency hint;
    # unrelated missing modules should surface unchanged for easier debugging.
    if exc.name != "isaaclab_ppisp" and not (exc.name and exc.name.startswith("isaaclab_ppisp.")):
        raise exc
    raise ModuleNotFoundError(_PPISP_IMPORT_ERROR_MESSAGE, name="isaaclab_ppisp") from exc


def _read_gpu_transforms_enabled() -> bool:
    """Return whether OVRTX should read GPU transforms from its internal transform cache."""
    value = os.environ.get(_READ_GPU_TRANSFORMS_ENV, "1").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            f"Invalid value for environment variable `{_READ_GPU_TRANSFORMS_ENV}`: {value}. Expected 0 or 1."
        )
    return value == "1"


def _gpu_side_render_var_sync_enabled() -> bool:
    """Return whether a render-var mapping is ordered by a GPU-side wait rather than a host wait.

    See :meth:`OVRTXRenderer._map_render_var_to_dlpack` for why Linux is the exception, and
    :data:`_DISABLE_LINUX_CUDA_CPU_SYNC_ENV` for opting out of it.

    Raises:
        ValueError: If the environment variable is set to anything other than ``0`` or ``1``.
    """
    if not sys.platform.startswith("linux"):
        return True
    value = os.environ.get(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(
            f"Invalid value for environment variable `{_DISABLE_LINUX_CUDA_CPU_SYNC_ENV}`: {value}. Expected 0 or 1."
        )
    return value == "1"


def _resolve_rtx_minimal_mode(data_types: list[str]) -> int | None:
    """Resolve the RTX minimal mode from data types.

    RTX minimal mode is used to control the rendering quality. The higher the mode, the higher the quality.

    If multiple simple shading data types are requested, the first one in the list is used and a warning is logged.

    If no simple shading data types are requested, None is returned.

    Args:
        data_types: List of data types.

    Returns:
        The resolved RTX minimal mode if simple shading data types are requested, otherwise None.
    """
    filtered_data_types = [data_type for data_type in data_types if data_type in _RTX_MINIMAL_MODES]
    if not filtered_data_types:
        return None

    if len(filtered_data_types) > 1:
        logger.warning(
            "Multiple simple shading data types requested (%s). Using the first in the list (%s).",
            filtered_data_types,
            filtered_data_types[0],
        )

    return _RTX_MINIMAL_MODES[filtered_data_types[0]]


def _write_file(output_dir: Path, file_name: str, content: str) -> None:
    """Write ``content`` to ``output_dir / file_name``.

    Creates ``output_dir`` and any missing parents when needed.

    Args:
        output_dir: Directory that receives the file.
        file_name: Base name of the file to write.
        content: Text content written with UTF-8 encoding.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(content)
        logger.info("Wrote USD file: %s", output_path)


class OVRTXRenderData:
    """OVRTX-specific RenderData. Holds warp output buffers sized from :class:`CameraRenderSpec`."""

    def __init__(self, spec: CameraRenderSpec, device):
        """Create render data from a camera render specification."""
        self.width = spec.cfg.width
        self.height = spec.cfg.height
        self.num_envs = spec.num_instances
        self.data_types = spec.cfg.data_types if spec.cfg.data_types else ["rgb"]
        self.num_cols = math.ceil(math.sqrt(self.num_envs))
        self.num_rows = math.ceil(self.num_envs / self.num_cols)
        self.warp_buffers: dict[str, wp.array] = {}
        # Per-output metadata collected during render() and copied into CameraData.info by read_output().
        # Populated for "semantic_segmentation" (with an "idToLabels" mapping) and
        # "instance_segmentation" (with "idToLabels" and "idToSemantics" mappings).
        self.renderer_info: dict[str, Any] = {}
        # Post-render PPISP pipeline composed when ``spec.cfg.isp_cfg`` is set.
        # ``isp_cfg`` is already fully normalized by ``prepare_cameras`` by the time it reaches here.
        self.ppisp_pipeline: PpispPipeline | None = None
        if spec.cfg.isp_cfg is not None:
            try:
                from isaaclab_ppisp import PpispPipeline
            except ModuleNotFoundError as exc:
                _raise_missing_ppisp_error(exc)

            self.ppisp_pipeline = PpispPipeline(spec.cfg.isp_cfg)


class OVRTXRenderer(BaseRenderer):
    """OVRTX Renderer implementation using the ovrtx library.

    This renderer uses the ovrtx library for high-fidelity RTX-based rendering,
    providing ray-traced rendering capabilities for Isaac Lab environments.
    """

    cfg: OVRTXRendererCfg

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output layout this OVRTX backend writes.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.supported_output_types`."""
        instance_seg_spec = (
            RenderBufferSpec(4, wp.uint8) if self.cfg.colorize_instance_segmentation else RenderBufferSpec(1, wp.int32)
        )
        semantic_seg_spec = (
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
            RenderBufferKind.SEMANTIC_SEGMENTATION: semantic_seg_spec,
            RenderBufferKind.INSTANCE_SEGMENTATION: instance_seg_spec,
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.MOTION_VECTORS: RenderBufferSpec(2, wp.float32),
        }

    def __init__(self, cfg: OVRTXRendererCfg):
        self.cfg = cfg
        self._device = "cuda:0"
        self._render_product_paths = []
        self._object_newton_indices: wp.array | None = None
        self._object_scales: wp.array | None = None
        self._object_scales_by_path: dict[str, tuple[float, float, float]] = {}
        self._deformable_particle_offsets: list[int] = []
        self._deformable_particle_counts: list[int] = []
        self._particle_visual_offsets: list[int] = []
        self._particle_visual_counts: list[int] = []
        self._cable_segment_counts: list[int] = []
        self._cable_max_points: int = 0
        self._cable_shape_ids: wp.array | None = None
        self._cable_offsets: wp.array | None = None
        self._cable_counts: wp.array | None = None
        self._cable_points: wp.array | None = None
        self._initialized_scene = False
        self._exported_usd_string: str | None = None
        self._camera_rel_path: str | None = None
        self._output_id_color_buffers: dict[str, wp.array] = {}
        self._clone_plan: ClonePlan | None = None
        self._visual_material_writer_ref: weakref.ReferenceType[OVRTXVisualMaterialWriter] | None = None
        self._use_ovstage = ovrtx_use_ovstage_enabled()
        self._init_fields()

        logger.info("Creating OVRTX renderer...")
        config = RendererConfig(
            log_file_path=self.cfg.log_file_path,
            log_level=self.cfg.log_level,
            read_gpu_transforms=_read_gpu_transforms_enabled(),
            keep_system_alive=True,
            suppress_deprecation_warnings=True,
            texture_streaming_mode=TextureStreamingMode.SYNCHRONOUS,
        )
        self._renderer = Renderer(config)
        if not self._renderer:
            raise RuntimeError(
                "Failed to create OVRTX Renderer; the underlying ovrtx.Renderer constructor returned a falsy"
                " value. Check that ovrtx is installed correctly and its native dependencies are available."
            )
        logger.info("OVRTX renderer created successfully")

    def _create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> OVRTXVisualMaterialWriter:
        if not self._initialized_scene:
            raise RuntimeError("OVRTX must ingest its detached scene before material writes are compiled.")
        writer = OVRTXVisualMaterialWriter(self, batches)
        self._visual_material_writer_ref = weakref.ref(writer)
        return writer

    @property
    def visual_material_writer(self):
        """Return the detached-scene material-writer factory."""
        return self._create_visual_material_writer

    def prepare_cameras(self, stage: Any, spec: CameraRenderSpec) -> None:
        if spec.cfg.isp_cfg is None:
            return
        try:
            from isaaclab_ppisp import apply_rtx_exposure_overrides, resolve_and_normalize
        except ModuleNotFoundError as exc:
            _raise_missing_ppisp_error(exc)
        camera_prim_path = spec.camera_prim_paths[0] if spec.camera_prim_paths else None
        spec.cfg.isp_cfg = resolve_and_normalize(spec.cfg.isp_cfg, stage, camera_prim_path)
        if spec.cfg.isp_cfg is None or not spec.camera_prim_paths:
            return
        apply_rtx_exposure_overrides(stage, list(spec.camera_prim_paths))

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        if stage is None:
            return
        self._clone_plan = SimulationContext.instance().get_clone_plan()
        if self._clone_plan is None or self._clone_plan.env_ids is None or self._clone_plan.positions is None:
            raise RuntimeError("Clone plan with environment ids and positions is required when preparing OVRTX stage")
        expected_ids = torch.arange(num_envs, device=self._clone_plan.env_ids.device)
        if not torch.equal(self._clone_plan.env_ids, expected_ids):
            raise RuntimeError("OVRTX requires ClonePlan environment ids ordered from zero.")
        if self.cfg.temp_usd_dir is not None:
            _write_file(Path(self.cfg.temp_usd_dir), "pre_ovrtx_renderer_stage.usda", stage.ExportToString())
        logger.info("Preparing stage (%d envs)...", num_envs)
        create_scene_partition_attributes(stage, num_envs)
        self._capture_object_scales(stage)
        self._exported_usd_string = export_stage_to_string(
            stage,
            num_envs,
            source_paths=self._clone_plan.sources,
            keep_env_roots=not self._use_ovstage,
        )

    def _capture_object_scales(self, stage: Any) -> None:
        self._object_scales_by_path.clear()
        from pxr import Gf, Usd, UsdGeom

        envs_prim = stage.GetPrimAtPath("/World/envs")
        if not envs_prim.IsValid():
            return
        xform_cache = UsdGeom.XformCache()
        for prim in Usd.PrimRange(envs_prim):
            if not prim.IsA(UsdGeom.Xformable):
                continue
            scale = Gf.Transform(xform_cache.GetLocalToWorldTransform(prim)).GetScale()
            scale = (float(scale[0]), float(scale[1]), float(scale[2]))
            if not all(math.isclose(axis, 1.0, rel_tol=1e-6, abs_tol=1e-6) for axis in scale):
                self._object_scales_by_path[str(prim.GetPath())] = scale

    def _create_object_scale_array(self, object_paths: list[str]) -> wp.array:
        scales = [self._object_scales_by_path.get(path, (1.0, 1.0, 1.0)) for path in object_paths]
        return wp.array(scales, dtype=wp.vec3f, device=self._device)

    def _init_fields_legacy(self) -> None:
        self._camera_xform_binding = None
        self._object_xform_binding = None
        self._deformable_points_binding = None
        self._particle_points_binding = None
        self._particle_workaround_applied = False
        self._cable_points_binding = None
        self._cable_point_slices: list[wp.array] = []

    def _initialize_from_spec_legacy(self, spec: CameraRenderSpec):
        width = spec.cfg.width
        height = spec.cfg.height
        num_envs = spec.num_instances
        data_types = spec.cfg.data_types if spec.cfg.data_types else ["rgb"]
        if spec.cfg.isp_cfg is not None and "rgb_hdr" not in data_types:
            data_types = [*data_types, "rgb_hdr"]
        env_0_prefix = "/World/envs/env_0/"
        first_cam_path = spec.camera_prim_paths[0]
        if not first_cam_path.startswith(env_0_prefix):
            raise RuntimeError(f"Expected camera prim under '{env_0_prefix}', got '{first_cam_path}'")
        self._camera_rel_path = spec.camera_path_relative_to_env_0
        if self._exported_usd_string is None:
            raise RuntimeError("Expected an exported USD string from stage")
        render_product_string, render_product_path = build_render_product_as_string(
            width=width,
            height=height,
            num_envs=num_envs,
            data_types=data_types,
            minimal_mode=_resolve_rtx_minimal_mode(data_types),
            camera_rel_path=self._camera_rel_path,
            background_color=getattr(spec.cfg, "background_color", None),
            device_id=cuda_device_id(self._device),
            enable_shadows=self.cfg.enable_shadows,
        )
        self._render_product_paths.append(render_product_path)
        combined_usd_string = self._exported_usd_string + "\n\n" + render_product_string
        self._exported_usd_string = None
        if self.cfg.temp_usd_dir is not None:
            _write_file(Path(self.cfg.temp_usd_dir), "ovrtx_renderer_stage.usda", combined_usd_string)
        self._renderer.open_usd_from_string(combined_usd_string)
        camera_paths = [f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(num_envs)]
        if num_envs > 1:
            self._clone_sources_in_ovrtx()
            self._update_scene_partitions_after_clone(num_envs)
            self._renderer.write_array_attribute(
                prim_paths=[render_product_path], attribute_name="camera", tensors=[camera_paths]
            )
        self._initialized_scene = True
        self._camera_xform_binding = self._renderer.bind_attribute(
            prim_paths=camera_paths,
            attribute_name="omni:xform",
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.EXISTING_ONLY,
        )
        self._renderer.write_attribute(
            prim_paths=camera_paths,
            attribute_name="omni:resetXformStack",
            tensor=np.full(num_envs, True, dtype=np.bool_),
        )
        if self._camera_xform_binding is None:
            raise RuntimeError("Camera binding is None — cannot render without a valid camera binding")
        self._setup_xform_bindings()
        self._setup_deformable_bindings(num_envs)
        self._setup_particle_bindings()
        self._setup_cable_bindings()

    def _clone_sources_in_ovrtx(self):
        clone_plan = self._clone_plan
        if clone_plan is None or clone_plan.env_ids is None or clone_plan.positions is None:
            raise RuntimeError("Clone plan with environment ids and positions is required when using OVRTX cloning")
        env_ids = clone_plan.env_ids.detach().cpu()
        clone_mask = clone_plan.clone_mask.detach().cpu()
        num_envs = len(env_ids)
        env_prim_paths = [f"/World/envs/env_{env_id}" for env_id in env_ids.tolist()]
        for row_idx, (source, destination) in enumerate(zip(clone_plan.sources, clone_plan.destinations, strict=True)):
            target_paths = [
                destination.format(int(env_id))
                for env_id in env_ids[clone_mask[row_idx]].tolist()
                if destination.format(int(env_id)) != source
            ]
            if target_paths:
                try:
                    self._renderer.clone_usd(source, target_paths)
                except Exception as exc:
                    raise RuntimeError(f"Failed to clone row {row_idx} from {source}: {exc}") from exc
        env_root_xforms = np.tile(np.eye(4, dtype=np.float64), (num_envs, 1, 1))
        env_root_xforms[:, 3, :3] = clone_plan.positions.cpu().numpy()
        self._renderer.write_attribute(
            prim_paths=env_prim_paths,
            attribute_name="omni:xform",
            tensor=env_root_xforms,
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.MUST_EXIST,
        )

    def _update_scene_partitions_after_clone(self, num_envs: int):
        partition_tokens = [f"env_{i}" for i in range(num_envs)]
        env_prim_paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
        camera_prim_paths = [f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(num_envs)]
        self._renderer.write_attribute(
            env_prim_paths, "primvars:omni:scenePartition", partition_tokens, semantic=Semantic.TOKEN_STRING
        )
        self._renderer.write_attribute(
            camera_prim_paths, "omni:scenePartition", partition_tokens, semantic=Semantic.TOKEN_STRING
        )

    def _setup_xform_bindings_legacy(self):
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return
        if SimulationContext.instance() is None:
            return
        newton_model = NewtonManager.get_model()
        if newton_model is None:
            return
        all_body_paths = getattr(newton_model, "body_label", None)
        if all_body_paths is None:
            return
        object_paths = []
        newton_indices = []
        for idx, path in enumerate(all_body_paths):
            if "/World/envs/" in path and self._camera_rel_path not in path and "GroundPlane" not in path:
                object_paths.append(path)
                newton_indices.append(idx)
        if not object_paths:
            return
        self._object_xform_binding = self._renderer.bind_attribute(
            prim_paths=object_paths,
            attribute_name="omni:xform",
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.EXISTING_ONLY,
        )
        self._renderer.write_attribute(
            prim_paths=object_paths,
            attribute_name="omni:resetXformStack",
            tensor=np.full(len(object_paths), True, dtype=np.bool_),
        )
        if self._object_xform_binding is None:
            raise RuntimeError("Failed to create OVRTX object bindings")
        self._object_newton_indices = wp.array(newton_indices, dtype=wp.int32, device=self._device)
        self._object_scales = self._create_object_scale_array(object_paths)

    def _setup_deformable_bindings_legacy(self, num_envs: int):
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return
        deformable_registry = NewtonManager._deformable_registry
        if not deformable_registry:
            return
        bad_entries = [entry for entry in deformable_registry if len(entry.particle_offsets) != num_envs]
        if bad_entries:
            details = "\n".join(
                f"- '{entry.prim_path}' has {len(entry.particle_offsets)} particle offsets" for entry in bad_entries
            )
            raise RuntimeError(
                f"OVRTX expects one particle offset per environment ({num_envs}), but the following "
                f"deformable entries have a mismatched offset count:\n{details}"
            )
        self._deformable_particle_offsets = []
        self._deformable_particle_counts = []
        vis_mesh_prim_paths = []
        for entry in deformable_registry:
            for idx, particle_offset in enumerate(entry.particle_offsets):
                self._deformable_particle_offsets.append(particle_offset)
                self._deformable_particle_counts.append(entry.particles_per_body)
                vis_mesh_prim_paths.append(
                    re.sub(r"(?<=[Ee]nv_)(?:\[\^/\][*+]|\.\*)", str(idx), entry.vis_mesh_prim_path)
                )
        prim_count = len(vis_mesh_prim_paths)
        if not prim_count:
            return
        self._renderer.write_attribute(
            prim_paths=vis_mesh_prim_paths,
            attribute_name="omni:resetXformStack",
            tensor=np.full(prim_count, True, dtype=np.bool_),
            prim_mode=PrimMode.MUST_EXIST,
        )
        self._renderer.write_attribute(
            prim_paths=vis_mesh_prim_paths,
            attribute_name="omni:xform",
            tensor=np.tile(np.eye(4, dtype=np.float64), (prim_count, 1, 1)),
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.MUST_EXIST,
        )
        self._deformable_points_binding = self._renderer.bind_array_attribute(
            prim_paths=vis_mesh_prim_paths,
            attribute_name="points",
            dtype=np.float32,
            shape=(3,),
            prim_mode=PrimMode.MUST_EXIST,
            flags=BindingFlag.OPTIMIZE,
        )
        if self._deformable_points_binding is None:
            raise RuntimeError("Failed to create OVRTX deformable body bindings")

    def _setup_cable_bindings_legacy(self) -> None:
        discovered = self._discover_cable_segment_bindings()
        if discovered is None:
            return
        cable_prim_paths, flat_shape_ids, offsets, counts = discovered
        prim_count = len(cable_prim_paths)
        self._renderer.write_attribute(
            prim_paths=cable_prim_paths,
            attribute_name="omni:resetXformStack",
            tensor=np.full(prim_count, True, dtype=np.bool_),
            prim_mode=PrimMode.MUST_EXIST,
        )
        self._renderer.write_attribute(
            prim_paths=cable_prim_paths,
            attribute_name="omni:xform",
            tensor=np.tile(np.eye(4, dtype=np.float64), (prim_count, 1, 1)),
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.MUST_EXIST,
        )
        self._cable_points_binding = self._renderer.bind_array_attribute(
            prim_paths=cable_prim_paths,
            attribute_name="points",
            dtype=np.float32,
            shape=(3,),
            prim_mode=PrimMode.MUST_EXIST,
            flags=BindingFlag.OPTIMIZE,
        )
        if self._cable_points_binding is None:
            raise RuntimeError("Failed to create OVRTX cable point bindings")
        self._allocate_cable_device_buffers(flat_shape_ids, offsets, counts)
        self._cable_point_slices = [
            self._cable_points[offset + curve : offset + curve + segment_count + 1]
            for curve, (offset, segment_count) in enumerate(zip(offsets, counts, strict=True))
        ]

    def _setup_particle_bindings_legacy(self) -> None:
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return
        particle_visual_prims = NewtonManager._particle_visual_prims
        if not particle_visual_prims:
            return
        self._particle_visual_offsets = []
        self._particle_visual_counts = []
        points_prim_paths = []
        for prim_path, record in particle_visual_prims.items():
            points_prim_paths.append(prim_path)
            self._particle_visual_offsets.append(record.offset)
            self._particle_visual_counts.append(record.count)
        prim_count = len(points_prim_paths)
        self._renderer.write_attribute(
            prim_paths=points_prim_paths,
            attribute_name="omni:resetXformStack",
            tensor=np.full(prim_count, True, dtype=np.bool_),
            prim_mode=PrimMode.MUST_EXIST,
        )
        self._renderer.write_attribute(
            prim_paths=points_prim_paths,
            attribute_name="omni:xform",
            tensor=np.tile(np.eye(4, dtype=np.float64), (prim_count, 1, 1)),
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.MUST_EXIST,
        )
        self._particle_points_binding = self._renderer.bind_array_attribute(
            prim_paths=points_prim_paths,
            attribute_name="points",
            dtype=np.float32,
            shape=(3,),
            prim_mode=PrimMode.MUST_EXIST,
            flags=BindingFlag.OPTIMIZE,
        )

    def create_render_data(self, spec: CameraRenderSpec) -> OVRTXRenderData:
        self._device = spec.device
        if not self._initialized_scene:
            self._initialize_from_spec(spec)
        return OVRTXRenderData(spec, self._device)

    def set_outputs(self, render_data: OVRTXRenderData, output_data: dict[str, ProxyArray]) -> None:
        render_data.warp_buffers = {
            name: proxy.warp for name, proxy in output_data.items() if name != str(RenderBufferKind.RGB)
        }
        if render_data.ppisp_pipeline is not None and str(RenderBufferKind.RGB_HDR) not in render_data.warp_buffers:
            ref_proxy = next(iter(output_data.values()))
            render_data.warp_buffers[str(RenderBufferKind.RGB_HDR)] = wp.zeros(
                (render_data.num_envs, render_data.height, render_data.width, 3),
                dtype=wp.float32,
                device=ref_proxy.device,
            )
        if render_data.ppisp_pipeline is not None and str(RenderBufferKind.RGBA) not in render_data.warp_buffers:
            raise ValueError(
                "OVRTX renderer ISP requires 'rgba' (or 'rgb', which aliases into rgba) as the"
                " LDR output destination, but neither was provided. Add 'rgb' or 'rgba' to"
                " Camera.cfg.data_types when isp_cfg is set."
            )

    def _update_transforms_legacy(self) -> None:
        if self._object_xform_binding is None or self._object_newton_indices is None or self._object_scales is None:
            return
        from isaaclab_newton.physics import NewtonManager

        newton_state = NewtonManager.get_state()
        if newton_state is None:
            raise RuntimeError("Newton state should not be None")
        body_q = getattr(newton_state, "body_q", None)
        if body_q is None:
            return
        with map_attribute_for_warp_writes(self._object_xform_binding, self._device, wp.mat44d) as ovrtx_transforms:
            wp.launch(
                kernel=sync_newton_transforms_kernel,
                dim=len(self._object_newton_indices),
                inputs=[ovrtx_transforms, self._object_newton_indices, body_q, self._object_scales],
                device=self._device,
            )

    def _update_geometries_legacy(self) -> None:
        if self._deformable_points_binding is not None:
            self._write_particle_q_slices(
                self._deformable_points_binding,
                self._deformable_particle_offsets,
                self._deformable_particle_counts,
            )
        if self._particle_points_binding is not None:
            self._write_particle_q_slices(
                self._particle_points_binding,
                self._particle_visual_offsets,
                self._particle_visual_counts,
            )
        if self._cable_points_binding is not None:
            self._write_cable_points_legacy()

    def _write_cable_points_legacy(self) -> None:
        self._compute_cable_points_world()
        self._cable_points_binding.write(
            cast(Any, self._cable_point_slices),
            data_access=DataAccess.ASYNC,
            cuda_stream=wp.get_stream(self._device).cuda_stream,
        )

    def _write_particle_q_slices(self, binding: Any, particle_offsets: list[int], particle_counts: list[int]) -> None:
        from isaaclab_newton.physics import NewtonManager

        state = NewtonManager.get_state()
        if state is None:
            raise RuntimeError("Newton state should not be None")
        particle_q = getattr(state, "particle_q", None)
        if particle_q is None:
            raise RuntimeError("Newton state has no particle_q but particle geometry bindings exist")
        particle_slices = [
            particle_q[offset : offset + count] for offset, count in zip(particle_offsets, particle_counts, strict=True)
        ]
        binding.write(
            cast(Any, particle_slices),
            data_access=DataAccess.ASYNC,
            cuda_stream=wp.get_stream(self._device).cuda_stream,
        )

    def _update_camera_legacy(
        self, render_data: OVRTXRenderData, positions: ProxyArray, orientations: ProxyArray, intrinsics: ProxyArray
    ) -> None:
        num_envs = positions.shape[0]
        converted_wp = wp.empty(num_envs, dtype=wp.quatf, device=self._device)
        convert_camera_frame_orientation_convention_wp(
            src=orientations.warp,
            dst=converted_wp,
            origin="world",
            target="opengl",
            device=self._device,
        )
        camera_transforms = wp.zeros(num_envs, dtype=wp.mat44d, device=self._device)
        wp.launch(
            kernel=create_camera_transforms_kernel,
            dim=num_envs,
            inputs=[positions, converted_wp, camera_transforms],
            device=self._device,
        )
        if self._camera_xform_binding is not None:
            with map_attribute_for_warp_writes(self._camera_xform_binding, self._device, wp.mat44d) as transforms_view:
                wp.copy(transforms_view, camera_transforms)

    def read_output(self, render_data: OVRTXRenderData, camera_data: CameraData) -> None:
        assert camera_data.info is not None, "CameraData.info should be created in CameraData.allocate"
        for output_name in camera_data.info:
            camera_data.info[output_name] = render_data.renderer_info.get(output_name)

    def _generate_random_colors_from_ids(self, input_ids: wp.array, output_colors: wp.array | None) -> wp.array:
        if output_colors is None or output_colors.shape != input_ids.shape:
            output_colors = wp.zeros(shape=input_ids.shape, dtype=wp.uint32, device=self._device)
        wp.launch(
            kernel=generate_random_colors_from_ids_kernel,
            dim=input_ids.shape,
            inputs=[input_ids, output_colors],
            device=self._device,
        )
        return output_colors

    @contextlib.contextmanager
    def _map_render_var_to_dlpack(self, render_var: Any) -> Iterator[wp.array]:
        gpu_side_sync = _gpu_side_render_var_sync_enabled()
        sync_stream = wp.get_stream(self._device).cuda_stream if gpu_side_sync else 0
        with render_var.map(device=Device.CUDA, sync_stream=sync_stream) as mapping:
            if not gpu_side_sync:
                mapping.wait()
            yield wp.from_dlpack(mapping)

    def _process_id_segmentation_render_var(
        self,
        render_data: OVRTXRenderData,
        frame,
        output_buffers: dict,
        render_var_name: str,
        buffer_key: str,
        colorize: bool,
    ) -> None:
        if render_var_name not in frame.render_vars or buffer_key not in output_buffers:
            return
        with self._map_render_var_to_dlpack(frame.render_vars[render_var_name]) as tiled_data:
            if tiled_data.dtype != wp.uint32:
                return
            if colorize:
                color_buffer = self._generate_random_colors_from_ids(
                    tiled_data, self._output_id_color_buffers.get(buffer_key)
                )
                self._output_id_color_buffers[buffer_key] = color_buffer
                colors_torch = wp.to_torch(color_buffer)
                colors_uint8 = colors_torch.view(torch.uint8)
                if colors_torch.dim() == 2:
                    h, w = colors_torch.shape
                    colors_uint8 = colors_uint8.reshape(h, w, 4)
                tiled_data = wp.from_torch(colors_uint8, dtype=wp.uint8)
                self._extract_rgba_tiles(render_data, tiled_data, output_buffers, buffer_key)
            else:
                if tiled_data.ndim == 2:
                    tiled_data = tiled_data.reshape((*tiled_data.shape, 1))
                self._launch_extract_all_tiles(render_data, tiled_data, output_buffers[buffer_key])

    def _process_semantic_id_map(self, render_data: OVRTXRenderData, frame) -> None:
        if "SemanticIdMap" not in frame.render_vars:
            return
        with frame.render_vars["SemanticIdMap"].map(device=Device.CPU) as mapping:
            labels_by_id = decode_semantic_id_map(np.from_dlpack(mapping))
        render_data.renderer_info["semantic_segmentation"] = {
            "idToLabels": build_semantic_id_to_labels(
                labels_by_id, colorize=self.cfg.colorize_semantic_segmentation, device=self._device
            )
        }

    def _process_instance_segmentation_maps(self, render_data: OVRTXRenderData, frame) -> None:
        required_vars = ("StableIdSemanticIdMap", "StableIdMap", "SemanticIdMap")
        missing = [var for var in required_vars if var not in frame.render_vars]
        if missing:
            raise RuntimeError(
                f"instance_segmentation was requested but the following render vars are missing from the "
                f"OVRTX frame: {missing}. Available vars: {list(frame.render_vars.keys())}"
            )
        with frame.render_vars["StableIdSemanticIdMap"].map(device=Device.CPU) as mapping:
            stable_id_semantic_id_map = decode_stable_id_semantic_id_map(np.from_dlpack(mapping))
        with frame.render_vars["StableIdMap"].map(device=Device.CPU) as mapping:
            stable_id_to_path = decode_stable_id_map(np.from_dlpack(mapping))
        with frame.render_vars["SemanticIdMap"].map(device=Device.CPU) as mapping:
            semantic_id_to_labels = decode_semantic_id_map(np.from_dlpack(mapping))
        id_to_labels, id_to_semantics = build_instance_id_to_labels_and_semantics(
            stable_id_semantic_id_map,
            stable_id_to_path,
            semantic_id_to_labels,
            colorize=self.cfg.colorize_instance_segmentation,
            device=self._device,
        )
        render_data.renderer_info["instance_segmentation"] = {
            "idToLabels": id_to_labels,
            "idToSemantics": id_to_semantics,
        }

    def _launch_extract_all_tiles(
        self, render_data: OVRTXRenderData, tiled_buffer: wp.array, output_buffer: wp.array
    ) -> None:
        tiled_channels = tiled_buffer.shape[-1]
        output_channels = output_buffer.shape[-1]
        if output_channels > tiled_channels:
            raise ValueError(
                f"Output buffer has {output_channels} channels but the tiled buffer only has {tiled_channels};"
                " extract_all_tiles_kernel would read out of bounds."
            )
        wp.launch(
            kernel=extract_all_tiles_kernel,
            dim=(render_data.num_envs, render_data.height, render_data.width),
            inputs=[tiled_buffer, output_buffer, render_data.num_cols, render_data.width, render_data.height],
            device=self._device,
        )

    def _extract_rgba_tiles(
        self,
        render_data: OVRTXRenderData,
        tiled_data: wp.array,
        output_buffers: dict,
        buffer_key: str,
        suffix: str = "",
    ) -> None:
        output_buffer = output_buffers[buffer_key]
        if output_buffer.shape[-1] not in (3, 4):
            raise ValueError(f"Expected RGB (3 channels) or RGBA (4 channels), got {output_buffer.shape[-1]}")
        self._launch_extract_all_tiles(render_data, tiled_data, output_buffer)

    def _extract_depth_tiles(
        self,
        render_data: OVRTXRenderData,
        tiled_depth_data: wp.array,
        output_buffers: dict,
        buffer_keys: Sequence[str],
    ) -> None:
        for depth_type in buffer_keys:
            if depth_type in output_buffers:
                self._launch_extract_all_tiles(render_data, tiled_depth_data, output_buffers[depth_type])

    def _extract_hdr_color_tiles(
        self, render_data: OVRTXRenderData, tiled_data: wp.array, output_buffers: dict
    ) -> None:
        if "rgb_hdr" not in output_buffers:
            return
        if tiled_data.dtype not in (wp.float16, wp.float32):
            raise TypeError(f"Unsupported OVRTX HdrColor dtype: {tiled_data.dtype}.")
        self._launch_extract_all_tiles(render_data, tiled_data, output_buffers["rgb_hdr"])

    def _prepare_ppisp_hdr_source(
        self, render_data: OVRTXRenderData, tiled_data: wp.array, output_buffers: dict
    ) -> wp.array:
        if render_data.ppisp_pipeline is None:
            return tiled_data
        output_device = str(output_buffers[str(RenderBufferKind.RGB_HDR)].device)
        if str(tiled_data.device) == output_device:
            return tiled_data
        # The render product pins ``deviceIds`` to this renderer's CUDA device, so the mapping
        # normally lands on the output device already. This stays as a fallback for the case OVRTX
        # reports as "deviceIds ... not in the active device set" and falls back to automatic
        # assignment.
        return wp.clone(tiled_data, device=output_device)

    def _process_render_frame(self, render_data: OVRTXRenderData, frame, output_buffers: dict) -> None:
        render_data.renderer_info.clear()
        if "LdrColor" in frame.render_vars:
            buffer_key = None
            if render_data.ppisp_pipeline is None and "rgba" in output_buffers:
                buffer_key = "rgba"
            else:
                for data_type in _RTX_MINIMAL_MODES:
                    if data_type in output_buffers:
                        buffer_key = data_type
                        break
            if buffer_key is not None:
                with self._map_render_var_to_dlpack(frame.render_vars["LdrColor"]) as tiled_data:
                    self._extract_rgba_tiles(render_data, tiled_data, output_buffers, buffer_key)
        for depth_var, buffer_keys in _DEPTH_VAR_BUFFER_KEYS.items():
            if depth_var not in frame.render_vars or not any(key in output_buffers for key in buffer_keys):
                continue
            with self._map_render_var_to_dlpack(frame.render_vars[depth_var]) as tiled_depth_data:
                if tiled_depth_data.dtype == wp.uint32:
                    tiled_depth_data = wp.from_torch(
                        wp.to_torch(tiled_depth_data).view(torch.float32), dtype=wp.float32
                    )
                self._extract_depth_tiles(render_data, tiled_depth_data, output_buffers, buffer_keys)
        if "DiffuseAlbedoSD" in frame.render_vars and "albedo" in output_buffers:
            with self._map_render_var_to_dlpack(frame.render_vars["DiffuseAlbedoSD"]) as tiled_data:
                self._extract_rgba_tiles(render_data, tiled_data, output_buffers, "albedo", suffix="albedo")
        if "HdrColor" in frame.render_vars and "rgb_hdr" in output_buffers:
            with self._map_render_var_to_dlpack(frame.render_vars["HdrColor"]) as tiled_data:
                tiled_data = self._prepare_ppisp_hdr_source(render_data, tiled_data, output_buffers)
                self._extract_hdr_color_tiles(render_data, tiled_data, output_buffers)
        self._process_id_segmentation_render_var(
            render_data,
            frame,
            output_buffers,
            "SemanticSegmentation",
            "semantic_segmentation",
            self.cfg.colorize_semantic_segmentation,
        )
        if "semantic_segmentation" in output_buffers:
            self._process_semantic_id_map(render_data, frame)
        self._process_id_segmentation_render_var(
            render_data,
            frame,
            output_buffers,
            "NonStableInstanceSegmentation",
            "instance_segmentation",
            self.cfg.colorize_instance_segmentation,
        )
        if "instance_segmentation" in output_buffers:
            self._process_instance_segmentation_maps(render_data, frame)
        if "NormalSD" in frame.render_vars and "normals" in output_buffers:
            with self._map_render_var_to_dlpack(frame.render_vars["NormalSD"]) as tiled_data:
                self._launch_extract_all_tiles(render_data, tiled_data, output_buffers["normals"])
        if "TargetMotionSD" in frame.render_vars and "motion_vectors" in output_buffers:
            with self._map_render_var_to_dlpack(frame.render_vars["TargetMotionSD"]) as tiled_data:
                self._launch_extract_all_tiles(render_data, tiled_data, output_buffers["motion_vectors"])

    def _render_legacy(self, render_data: OVRTXRenderData) -> None:
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call initialize() first.")
        if self._renderer is None or not self._render_product_paths:
            return
        material_writer = self._visual_material_writer_ref() if self._visual_material_writer_ref is not None else None
        try:
            if material_writer is not None:
                material_writer.publish()
            products = self._renderer.step(render_products=set(self._render_product_paths), delta_time=1.0 / 60.0)
        finally:
            if material_writer is not None:
                drain_errors = contextlib.nullcontext() if sys.exc_info()[0] is None else contextlib.suppress(Exception)
                with drain_errors:
                    material_writer.drain()
        product_path = self._render_product_paths[0]
        if product_path in products and products[product_path].frames:
            self._process_render_frame(render_data, products[product_path].frames[0], render_data.warp_buffers)
        if render_data.ppisp_pipeline is not None:
            render_data.ppisp_pipeline.apply(
                render_data.warp_buffers[str(RenderBufferKind.RGB_HDR)],
                render_data.warp_buffers[str(RenderBufferKind.RGBA)],
            )

    def _close_legacy(self) -> None:
        def _safe_unbind(binding, name: str) -> None:
            if binding is None:
                return
            try:
                binding.unbind()
            except Exception as exc:
                if "destroyed" not in str(exc).lower():
                    logger.warning("Error unbinding %s: %s", name, exc)

        for attr, name in (
            ("_camera_xform_binding", "camera transforms"),
            ("_object_xform_binding", "object transforms"),
            ("_deformable_points_binding", "deformable points"),
            ("_particle_points_binding", "particle points"),
            ("_cable_points_binding", "cable points"),
        ):
            _safe_unbind(getattr(self, attr), name)
            setattr(self, attr, None)
        self._deformable_particle_offsets = []
        self._deformable_particle_counts = []
        self._particle_visual_offsets = []
        self._particle_visual_counts = []
        self._particle_workaround_applied = False
        self._cable_point_slices = []
        self._cable_segment_counts = []
        self._cable_max_points = 0
        self._cable_points = None
        self._cable_shape_ids = None
        self._cable_offsets = None
        self._cable_counts = None
        if self._renderer:
            try:
                self._renderer.reset_stage()
            except Exception as exc:
                logger.warning("Error resetting stage: %s", exc)
            self._renderer = None
        self._render_product_paths.clear()
        self._output_id_color_buffers.clear()
        self._initialized_scene = False

    def _init_fields(self) -> None:
        self._init_fields_ovstage() if self._use_ovstage else self._init_fields_legacy()

    def _initialize_from_spec(self, spec: CameraRenderSpec) -> None:
        self._initialize_from_spec_ovstage(spec) if self._use_ovstage else self._initialize_from_spec_legacy(spec)

    def _setup_xform_bindings(self) -> None:
        self._setup_xform_bindings_ovstage() if self._use_ovstage else self._setup_xform_bindings_legacy()

    def _setup_deformable_bindings(self, num_envs: int) -> None:
        self._setup_deformable_bindings_ovstage(
            num_envs
        ) if self._use_ovstage else self._setup_deformable_bindings_legacy(num_envs)

    def _setup_particle_bindings(self) -> None:
        self._setup_particle_bindings_ovstage() if self._use_ovstage else self._setup_particle_bindings_legacy()

    def _setup_cable_bindings(self) -> None:
        self._setup_cable_bindings_ovstage() if self._use_ovstage else self._setup_cable_bindings_legacy()

    @staticmethod
    def _discover_cable_segment_bindings() -> tuple[list[str], list[int], list[int], list[int]] | None:
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return None
        cable_segment_shape_ids = NewtonManager.collect_cable_segment_shape_ids()
        if not cable_segment_shape_ids:
            return None
        paths, flat_ids, offsets, counts = [], [], [], []
        for prim_path, segment_shape_ids in cable_segment_shape_ids.items():
            paths.append(prim_path)
            offsets.append(len(flat_ids))
            counts.append(len(segment_shape_ids))
            flat_ids.extend(segment_shape_ids)
        return paths, flat_ids, offsets, counts

    def _allocate_cable_device_buffers(self, flat_shape_ids: list[int], offsets: list[int], counts: list[int]) -> None:
        self._cable_shape_ids = wp.array(flat_shape_ids, dtype=wp.int32, device=self._device)
        self._cable_offsets = wp.array(offsets, dtype=wp.int32, device=self._device)
        self._cable_counts = wp.array(counts, dtype=wp.int32, device=self._device)
        self._cable_segment_counts = counts
        self._cable_max_points = max(counts) + 1 if counts else 0
        self._cable_points = wp.zeros(sum(count + 1 for count in counts), dtype=wp.vec3f, device=self._device)

    def _compute_cable_points_world(self) -> None:
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        state = NewtonManager.get_state()
        if state is None:
            raise RuntimeError("Newton state should not be None")
        wp.launch(
            compute_cable_points_world_kernel,
            dim=(len(self._cable_segment_counts), self._cable_max_points),
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

    def update_transforms(self) -> None:
        self._update_transforms_ovstage() if self._use_ovstage else self._update_transforms_legacy()

    def update_geometries(self) -> None:
        self._update_geometries_ovstage() if self._use_ovstage else self._update_geometries_legacy()

    def update_camera(
        self, render_data: OVRTXRenderData, positions: ProxyArray, orientations: ProxyArray, intrinsics: ProxyArray
    ) -> None:
        if self._use_ovstage:
            self._update_camera_ovstage(render_data, positions, orientations, intrinsics)
        else:
            self._update_camera_legacy(render_data, positions, orientations, intrinsics)

    def render(self, render_data: OVRTXRenderData) -> None:
        self._render_ovstage(render_data) if self._use_ovstage else self._render_legacy(render_data)

    def cleanup(self, render_data: OVRTXRenderData | None) -> None:
        if render_data is None:
            return
        render_data.warp_buffers.clear()
        render_data.renderer_info.clear()
        render_data.ppisp_pipeline = None

    def close(self) -> None:
        self._close_ovstage() if self._use_ovstage else self._close_legacy()
        self._visual_material_writer_ref = None

    def _init_fields_ovstage(self) -> None:
        self._stage = None
        self._stage_paths = None
        self._ovstage_exit_stack: contextlib.ExitStack | None = None
        self._current_ordinal = 0
        self._camera_xform_query = self._camera_paths_list = None
        self._object_xform_query = self._object_paths_list = None
        self._deformable_points_query = self._deformable_paths_list = None
        self._particle_points_query = self._particle_paths_list = None
        self._cable_points_query = self._cable_paths_list = None
        self._cable_point_tensors = []

    def _initialize_from_spec_ovstage(self, spec: CameraRenderSpec) -> None:
        width, height, num_envs = spec.cfg.width, spec.cfg.height, spec.num_instances
        data_types = spec.cfg.data_types if spec.cfg.data_types else ["rgb"]
        if spec.cfg.isp_cfg is not None and "rgb_hdr" not in data_types:
            data_types = [*data_types, "rgb_hdr"]
        first_cam_path = spec.camera_prim_paths[0]
        if not first_cam_path.startswith("/World/envs/env_0/"):
            raise RuntimeError(f"Expected camera prim under '/World/envs/env_0/', got '{first_cam_path}'")
        self._camera_rel_path = spec.camera_path_relative_to_env_0
        if self._exported_usd_string is None:
            raise RuntimeError("Expected an exported USD string from stage")
        render_product_string, render_product_path = build_render_product_as_string(
            width=width,
            height=height,
            num_envs=num_envs,
            data_types=data_types,
            minimal_mode=_resolve_rtx_minimal_mode(data_types),
            camera_rel_path=self._camera_rel_path,
            device_id=cuda_device_id(self._device),
            enable_shadows=self.cfg.enable_shadows,
        )
        self._render_product_paths.append(render_product_path)
        combined_usd_string = self._exported_usd_string + "\n\n" + render_product_string
        self._exported_usd_string = None
        if self.cfg.temp_usd_dir is not None:
            _write_file(Path(self.cfg.temp_usd_dir), "ovrtx_renderer_stage.usda", combined_usd_string)
        self._ovstage_exit_stack = contextlib.ExitStack()
        self._stage = self._ovstage_exit_stack.enter_context(create_ovstage("isaaclab.ovrtx"))
        self._stage_paths = self._ovstage_exit_stack.enter_context(ovstage.PathDictionary(self._stage))
        self._current_ordinal += 1
        ovstage.population.open_usd_from_string(
            self._stage,
            combined_usd_string,
            ordinal=self._current_ordinal,
            domains=ovstage.PopulationDomain.RENDERING,
        )
        if num_envs > 1:
            self._clone_sources_ovstage()
            self._update_scene_partitions_after_clone_ovstage(num_envs)
        self._initialized_scene = True
        camera_paths = [f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(num_envs)]
        render_product_paths = self._stage_paths.create_path_list_from_strings([render_product_path])
        with self._stage.query_from_path_list(render_product_paths) as query:
            target_ids = np.array([self._stage_paths.intern_path(path) for path in camera_paths], dtype=np.uint64)
            self._stage.write_attribute(
                query,
                self._stage_paths.intern_token("camera"),
                ordinal=self._current_ordinal,
                tensors=target_ids,
                is_array=True,
                semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
            ).wait()
        self._stage_paths.destroy_path_list(render_product_paths)
        self._camera_paths_list = self._stage_paths.create_path_list_from_strings(camera_paths)
        self._camera_xform_query = self._stage.query_from_path_list(self._camera_paths_list)
        if self._camera_xform_query is None:
            raise RuntimeError("Camera query is None — cannot render without a valid camera query")
        self._stage.write_attribute(
            self._camera_xform_query,
            "omni:resetXformStack",
            ordinal=self._current_ordinal,
            tensors=np.full(num_envs, True, dtype=np.bool_),
            is_array=False,
        ).wait()
        self._setup_xform_bindings_ovstage()
        self._setup_deformable_bindings_ovstage(num_envs)
        self._setup_particle_bindings_ovstage()
        self._setup_cable_bindings_ovstage()
        self._stage.advance_write_floor(ordinal=self._current_ordinal).wait()
        self._renderer.attach_ovstage(self._stage)
        self._current_ordinal += 1

    def _clone_sources_ovstage(self):
        clone_plan = self._clone_plan
        if clone_plan is None or clone_plan.env_ids is None or clone_plan.positions is None:
            raise RuntimeError("Clone plan with environment ids and positions is required when using OVRTX cloning")
        env_ids = clone_plan.env_ids.detach().cpu()
        clone_mask = clone_plan.clone_mask.detach().cpu()
        paths = [f"/World/envs/env_{env_id}" for env_id in env_ids.tolist()]
        for row_idx, (source, destination) in enumerate(zip(clone_plan.sources, clone_plan.destinations, strict=True)):
            targets = [
                destination.format(int(env_id))
                for env_id in env_ids[clone_mask[row_idx]].tolist()
                if destination.format(int(env_id)) != source
            ]
            if targets:
                self._stage.clone(source, targets, ordinal=self._current_ordinal)
        xforms = np.tile(np.eye(4, dtype=np.float64), (len(env_ids), 1, 1))
        xforms[:, 3, :3] = clone_plan.positions.cpu().numpy()
        path_list = self._stage_paths.create_path_list_from_strings(paths)
        query = self._stage.query_from_path_list(path_list)
        self._stage.write_attribute(
            query,
            "omni:xform",
            ordinal=self._current_ordinal,
            tensors=xform_tensor_from_numpy(xforms),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
        ).wait()
        self._stage.release_query(query).wait()
        self._stage_paths.destroy_path_list(path_list)

    def _update_scene_partitions_after_clone_ovstage(self, num_envs: int):
        tokens = np.array([self._stage_paths.intern_token(f"env_{i}") for i in range(num_envs)], dtype=np.uint64)
        for paths, attribute in (
            ([f"/World/envs/env_{i}" for i in range(num_envs)], "primvars:omni:scenePartition"),
            ([f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(num_envs)], "omni:scenePartition"),
        ):
            path_list = self._stage_paths.create_path_list_from_strings(paths)
            query = self._stage.query_from_path_list(path_list)
            self._stage.write_attribute(
                query,
                attribute,
                ordinal=self._current_ordinal,
                tensors=tokens,
                is_array=False,
                semantic=ovstage.AttributeSemantic.TOKEN_ID,
            ).wait()
            self._stage.release_query(query).wait()
            self._stage_paths.destroy_path_list(path_list)

    def _setup_xform_bindings_ovstage(self) -> None:
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return
        model = NewtonManager.get_model()
        if SimulationContext.instance() is None or model is None or getattr(model, "body_label", None) is None:
            return
        object_paths, indices = [], []
        for idx, path in enumerate(model.body_label):
            if "/World/envs/" in path and self._camera_rel_path not in path and "GroundPlane" not in path:
                object_paths.append(path)
                indices.append(idx)
        if not object_paths:
            return
        self._object_paths_list = self._stage_paths.create_path_list_from_strings(object_paths)
        self._object_xform_query = self._stage.query_from_path_list(self._object_paths_list)
        self._stage.write_attribute(
            self._object_xform_query,
            "omni:resetXformStack",
            ordinal=self._current_ordinal,
            tensors=np.full(len(object_paths), True, dtype=np.bool_),
            is_array=False,
        ).wait()
        self._object_newton_indices = wp.array(indices, dtype=wp.int32, device=self._device)
        self._object_scales = self._create_object_scale_array(object_paths)

    def _setup_deformable_bindings_ovstage(self, num_envs: int) -> None:
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return
        registry = NewtonManager._deformable_registry
        if not registry:
            return
        bad = [entry for entry in registry if len(entry.particle_offsets) != num_envs]
        if bad:
            raise RuntimeError(f"OVRTX expects one particle offset per environment ({num_envs})")
        paths = []
        self._deformable_particle_offsets = []
        self._deformable_particle_counts = []
        for entry in registry:
            for idx, offset in enumerate(entry.particle_offsets):
                paths.append(re.sub(r"(?<=[Ee]nv_)(?:\[\^/\][*+]|\.\*)", str(idx), entry.vis_mesh_prim_path))
                self._deformable_particle_offsets.append(offset)
                self._deformable_particle_counts.append(entry.particles_per_body)
        if not paths:
            return
        self._deformable_paths_list = self._stage_paths.create_path_list_from_strings(paths)
        self._deformable_points_query = self._stage.query_from_path_list(self._deformable_paths_list)
        self._write_identity_xforms(self._deformable_points_query, len(paths))

    def _setup_particle_bindings_ovstage(self) -> None:
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            return
        records = NewtonManager._particle_visual_prims
        if not records:
            return
        paths = list(records)
        self._particle_visual_offsets = [record.offset for record in records.values()]
        self._particle_visual_counts = [record.count for record in records.values()]
        self._particle_paths_list = self._stage_paths.create_path_list_from_strings(paths)
        self._particle_points_query = self._stage.query_from_path_list(self._particle_paths_list)
        self._write_identity_xforms(self._particle_points_query, len(paths))

    def _setup_cable_bindings_ovstage(self) -> None:
        discovered = self._discover_cable_segment_bindings()
        if discovered is None:
            return
        paths, ids, offsets, counts = discovered
        self._cable_paths_list = self._stage_paths.create_path_list_from_strings(paths)
        self._cable_points_query = self._stage.query_from_path_list(self._cable_paths_list)
        self._write_identity_xforms(self._cable_points_query, len(paths))
        self._allocate_cable_device_buffers(ids, offsets, counts)
        self._cable_point_slices = [
            self._cable_points[offset + curve : offset + curve + count + 1]
            for curve, (offset, count) in enumerate(zip(offsets, counts, strict=True))
        ]
        self._cable_point_tensors = [points_tensor_from_warp(points) for points in self._cable_point_slices]

    def _write_identity_xforms(self, query, count: int) -> None:
        self._stage.write_attribute(
            query,
            "omni:resetXformStack",
            ordinal=self._current_ordinal,
            tensors=np.full(count, True, dtype=np.bool_),
            is_array=False,
        ).wait()
        self._stage.write_attribute(
            query,
            "omni:xform",
            ordinal=self._current_ordinal,
            tensors=xform_tensor_from_numpy(np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
        ).wait()

    def _update_transforms_ovstage(self) -> None:
        if self._object_xform_query is None or self._object_newton_indices is None or self._object_scales is None:
            return
        from isaaclab_newton.physics import NewtonManager

        state = NewtonManager.get_state()
        if state is None:
            raise RuntimeError("Newton state should not be None")
        if getattr(state, "body_q", None) is None:
            return
        transforms = wp.empty(len(self._object_newton_indices), dtype=wp.mat44d, device=self._device)
        wp.launch(
            sync_newton_transforms_kernel,
            len(self._object_newton_indices),
            inputs=[transforms, self._object_newton_indices, state.body_q, self._object_scales],
            device=self._device,
        )
        self._stage.write_attribute(
            self._object_xform_query,
            "omni:xform",
            ordinal=self._current_ordinal,
            tensors=xform_tensor_from_warp(transforms),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            cuda_stream=wp.get_stream(self._device).cuda_stream,
        ).wait()

    def _update_geometries_ovstage(self) -> None:
        if self._deformable_points_query is not None or self._particle_points_query is not None:
            from isaaclab_newton.physics import NewtonManager

            state = NewtonManager.get_state()
            if state is None or getattr(state, "particle_q", None) is None:
                raise RuntimeError("Newton state has no particle_q but particle geometry queries exist")
            if self._deformable_points_query is not None:
                self._write_particle_q_slices_ovstage(
                    self._deformable_points_query,
                    state.particle_q,
                    self._deformable_particle_offsets,
                    self._deformable_particle_counts,
                )
            if self._particle_points_query is not None:
                self._write_particle_q_slices_ovstage(
                    self._particle_points_query,
                    state.particle_q,
                    self._particle_visual_offsets,
                    self._particle_visual_counts,
                )
        if self._cable_points_query is not None:
            self._write_cable_points_ovstage()

    def _write_particle_q_slices_ovstage(self, query, particle_q, offsets, counts) -> None:
        tensors = [
            points_tensor_from_warp(particle_q[offset : offset + count])
            for offset, count in zip(offsets, counts, strict=True)
        ]
        self._stage.write_attribute(
            query,
            "points",
            ordinal=self._current_ordinal,
            tensors=tensors,
            is_array=True,
            semantic=ovstage.AttributeSemantic.POINT,
            cuda_stream=wp.get_stream(self._device).cuda_stream,
        ).wait()

    def _write_cable_points_ovstage(self) -> None:
        self._compute_cable_points_world()
        self._stage.write_attribute(
            self._cable_points_query,
            "points",
            ordinal=self._current_ordinal,
            tensors=self._cable_point_tensors,
            is_array=True,
            semantic=ovstage.AttributeSemantic.POINT,
            cuda_stream=wp.get_stream(self._device).cuda_stream,
        ).wait()

    def _update_camera_ovstage(
        self, render_data: OVRTXRenderData, positions: ProxyArray, orientations: ProxyArray, intrinsics: ProxyArray
    ) -> None:
        num_envs = positions.shape[0]
        converted = wp.empty(num_envs, dtype=wp.quatf, device=self._device)
        convert_camera_frame_orientation_convention_wp(
            src=orientations.warp, dst=converted, origin="world", target="opengl", device=self._device
        )
        transforms = wp.zeros(num_envs, dtype=wp.mat44d, device=self._device)
        wp.launch(
            create_camera_transforms_kernel, num_envs, inputs=[positions, converted, transforms], device=self._device
        )
        if self._camera_xform_query is not None:
            self._stage.write_attribute(
                self._camera_xform_query,
                "omni:xform",
                ordinal=self._current_ordinal,
                tensors=xform_tensor_from_warp(transforms),
                is_array=False,
                semantic=ovstage.AttributeSemantic.MATRIX,
                cuda_stream=wp.get_stream(self._device).cuda_stream,
            ).wait()

    def _render_ovstage(self, render_data: OVRTXRenderData) -> None:
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call initialize() first.")
        if self._renderer is None or not self._render_product_paths:
            return
        writer = self._visual_material_writer_ref() if self._visual_material_writer_ref is not None else None
        try:
            if writer is not None:
                writer.publish()
            self._stage.advance_write_floor(ordinal=self._current_ordinal).wait()
        finally:
            if writer is not None:
                with contextlib.nullcontext() if sys.exc_info()[0] is None else contextlib.suppress(Exception):
                    writer.drain()
        products = self._renderer.step(
            render_products=set(self._render_product_paths),
            delta_time=1.0 / 60.0,
            ordinal=self._current_ordinal,
        )
        self._current_ordinal += 1
        path = self._render_product_paths[0]
        if path in products and products[path].frames:
            self._process_render_frame(render_data, products[path].frames[0], render_data.warp_buffers)
        if render_data.ppisp_pipeline is not None:
            render_data.ppisp_pipeline.apply(
                render_data.warp_buffers[str(RenderBufferKind.RGB_HDR)],
                render_data.warp_buffers[str(RenderBufferKind.RGBA)],
            )

    def _close_ovstage(self) -> None:
        def release(query, paths):
            if query is not None and self._stage is not None:
                with contextlib.suppress(Exception):
                    self._stage.release_query(query).wait()
            if paths is not None and self._stage_paths is not None:
                with contextlib.suppress(Exception):
                    self._stage_paths.destroy_path_list(paths)

        for query, paths in (
            (self._camera_xform_query, self._camera_paths_list),
            (self._object_xform_query, self._object_paths_list),
            (self._deformable_points_query, self._deformable_paths_list),
            (self._particle_points_query, self._particle_paths_list),
            (self._cable_points_query, self._cable_paths_list),
        ):
            release(query, paths)
        self._camera_xform_query = self._camera_paths_list = None
        self._object_xform_query = self._object_paths_list = None
        self._deformable_points_query = self._deformable_paths_list = None
        self._particle_points_query = self._particle_paths_list = None
        self._cable_points_query = self._cable_paths_list = None
        self._object_newton_indices = self._object_scales = None
        self._object_scales_by_path = {}
        self._deformable_particle_offsets = self._deformable_particle_counts = []
        self._particle_visual_offsets = self._particle_visual_counts = []
        self._cable_segment_counts = []
        self._cable_max_points = 0
        self._cable_point_tensors = self._cable_point_slices = []
        self._cable_points = self._cable_shape_ids = self._cable_offsets = self._cable_counts = None
        if self._renderer is not None and self._stage is not None:
            self._renderer.detach_ovstage()
        self._renderer = None
        if self._ovstage_exit_stack is not None:
            self._ovstage_exit_stack.close()
        self._ovstage_exit_stack = self._stage = self._stage_paths = None
        self._render_product_paths.clear()
        self._output_id_color_buffers.clear()
        self._initialized_scene = False
        self._current_ordinal = 0
