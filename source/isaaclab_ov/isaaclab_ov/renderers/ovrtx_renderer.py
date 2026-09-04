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

from isaaclab_ov.renderers.ovrtx_annotator_utils import (
    build_instance_id_to_labels_and_semantics,
    build_semantic_id_to_labels,
    decode_semantic_id_map,
    decode_stable_id_map,
    decode_stable_id_semantic_id_map,
)
from isaaclab_ov.renderers.ovrtx_compat import RENDER_VAR_FRAME_KEYS
from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXRendererCfg
from isaaclab_ov.renderers.ovrtx_renderer_kernels import (
    compute_cable_points_world_kernel,
    create_camera_transforms_kernel,
    extract_all_tiles_kernel,
    generate_random_colors_from_ids_kernel,
    sync_newton_transforms_kernel,
)
from isaaclab_ov.renderers.ovrtx_shader_cache import redirect_shader_cache
from isaaclab_ov.renderers.ovrtx_usd import (
    build_render_product_as_string,
    create_scene_partition_attributes,
    export_stage_to_string,
)
from isaaclab_ov.renderers.visual_materials import OVRTXVisualMaterialWriter
from isaaclab_ov.stage import (
    create_ovstage,
    points_tensor_from_warp,
    xform_tensor_from_numpy,
    xform_tensor_from_warp,
)

if TYPE_CHECKING:
    from isaaclab_ppisp import PpispPipeline

    from isaaclab.renderers.base_renderer import VisualMaterialBatch
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

from isaaclab.renderers.camera_render_spec import CameraRenderSpec

# ``frame.render_vars`` keys of the render vars read below. Baked at import from the installed
# OVRTX version, which decides whether frames are keyed by source name or RenderVar prim path.
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

# Map render vars needed to decode the instance-segmentation info dicts.
_INSTANCE_SEGMENTATION_MAP_VARS = (_STABLE_ID_SEMANTIC_ID_MAP_VAR, _STABLE_ID_MAP_VAR, _SEMANTIC_ID_MAP_VAR)

# Maps depth render vars to compatible output buffers.
_DEPTH_VAR_BUFFER_KEYS: dict[str, tuple[str, ...]] = {
    RENDER_VAR_FRAME_KEYS["DistanceToImagePlaneSD"]: ("depth", "distance_to_image_plane"),
    RENDER_VAR_FRAME_KEYS["DistanceToCameraSD"]: ("distance_to_camera",),
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
        # Semantic segmentation: colorized RGBA (uint8), else raw int32 IDs (matches Isaac RTX, whose
        # non-colorized per-pixel value is the semantic ID).
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
        self._device = "cuda:0"  # default; overridden by create_render_data(spec)
        # Resolved by create_render_data(spec); every render-product device id and CUDA sync stream
        # derives from this one cached device so a bare "cuda" cannot be re-interpreted per call site.
        self._warp_device: wp.Device | None = None
        self._render_product_paths = []
        # Shared by both paths. The legacy-only binding handles that pair with these live in
        # _init_fields_legacy instead; the ovstage path drives the same offsets and counts
        # through its stage queries.
        self._object_newton_indices: wp.array | None = None
        self._object_scales: wp.array | None = None
        self._object_scales_by_path: dict[str, tuple[float, float, float]] = {}
        self._deformable_particle_offsets: list[int] = []
        self._deformable_particle_counts: list[int] = []
        self._particle_visual_offsets: list[int] = []
        self._particle_visual_counts: list[int] = []
        # Shared Newton cable curve state used by both legacy and ovstage write paths.
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

        # Selected once at construction so every dispatch method below sees a stable path for the
        # lifetime of the renderer, even if the environment variable changes mid-process.
        self._use_ovstage = ovrtx_use_ovstage_enabled()
        self._init_fields()

        logger.info("Creating OVRTX renderer...")
        OVRTX_CONFIG = RendererConfig(
            log_file_path=self.cfg.log_file_path,
            log_level=self.cfg.log_level,
            read_gpu_transforms=_read_gpu_transforms_enabled(),
            keep_system_alive=True,
            suppress_deprecation_warnings=True,
            texture_streaming_mode=TextureStreamingMode.SYNCHRONOUS,
        )

        # Takes the config because the redirect can be what first loads the ovrtx
        # library, and initialization only happens once, so it has to see the
        # same config the renderer below is built with.
        redirect_shader_cache(OVRTX_CONFIG)

        self._renderer = Renderer(OVRTX_CONFIG)
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
        """Resolve the camera's PPISP cfg and apply OVRTX-specific USD overrides.

        When ``spec.cfg.isp_cfg`` is set, resolves it (sentinel discovery +
        normalization) via :func:`isaaclab_ppisp.resolve_and_normalize` so
        :mod:`isaaclab` does not need to know about PPISP. Then pins
        ``exposure:*`` to neutral and applies ``OmniRtxCameraExposureAPI_1`` so
        the RTX exposure model OVRTX embeds does not compound on top of the
        ISP. Without an ISP, the camera prim's authored exposure is left alone.
        """
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
        """Prepare the USD stage for OVRTX before :meth:`create_render_data`.

        Adds scene partition attributes and exports the stage to a string held on the renderer until
        :meth:`create_render_data` is called.
        """
        if stage is None:
            return

        self._clone_plan = SimulationContext.instance().get_clone_plan()
        if self._clone_plan is None or self._clone_plan.env_ids is None or self._clone_plan.positions is None:
            raise RuntimeError("Clone plan with environment ids and positions is required when preparing OVRTX stage")
        if not np.array_equal(self._clone_plan.env_ids, np.arange(num_envs)):
            raise RuntimeError("OVRTX requires ClonePlan environment ids ordered from zero.")

        # If temp_usd_dir is set, write the pre-ovrtx stage to a temporary file.
        if self.cfg.temp_usd_dir is not None:
            _write_file(Path(self.cfg.temp_usd_dir), "pre_ovrtx_renderer_stage.usda", stage.ExportToString())

        logger.info("Preparing stage (%d envs)...", num_envs)
        create_scene_partition_attributes(stage, num_envs)

        # Composed scales must be read while the full stage is still live, before export trims it.
        self._capture_object_scales(stage)

        # The clone plan already identifies every source row. Keep those rows independent so
        # backend bindings for dynamic assets retain the paths they were compiled against.
        self._exported_usd_string = export_stage_to_string(
            stage,
            num_envs,
            source_paths=self._clone_plan.sources,
            keep_env_roots=not self._use_ovstage,
        )

    def _capture_object_scales(self, stage: Any) -> None:
        """Record composed world scales of scaled environment prims before the stage is exported.

        The per-frame object transform write rebuilds each body's matrix from a Newton
        ``transformf``, which carries only translation and rotation, so any scale authored on the
        USD prim is lost once that write lands. Capturing the composed scale here, while the full
        stage is still live, lets :meth:`_create_object_scale_array` fold it back in.

        Only prims whose scale deviates from unit are stored, keeping the mapping small for scenes
        with many environments.

        Args:
            stage: The live USD stage, before per-environment trimming and export.
        """
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
        """Build the device scale array aligned with the Newton body binding order.

        Args:
            object_paths: Bound body prim paths, ordered to match the Newton index array.

        Returns:
            Per-body scale factors, shape ``[len(object_paths)]``, unit where no scale was authored.
        """
        scales = [self._object_scales_by_path.get(path, (1.0, 1.0, 1.0)) for path in object_paths]
        return wp.array(scales, dtype=wp.vec3f, device=self._device)

    def _init_fields_legacy(self) -> None:
        """Initialize the legacy-path instance fields.

        Counterpart to :meth:`_init_fields_ovstage`. Only fields the ovstage path never touches live
        here: the ``bind_attribute``/``bind_array_attribute`` handles and the caller-owned object
        transform buffer. State shared by both paths (``_object_newton_indices``, the particle
        offset/count lists) stays in :meth:`__init__`.
        """
        self._camera_xform_binding = None
        self._object_xform_binding = None
        self._object_transform_buffer: wp.array | None = None
        self._deformable_points_binding = None
        self._particle_points_binding = None
        self._particle_workaround_applied = False
        self._cable_points_binding = None
        # Stable Warp views into ``_cable_points`` for ASYNC GPU writes.
        self._cable_point_slices: list[wp.array] = []

    def _initialize_from_spec_legacy(self, spec: CameraRenderSpec):
        """Initialize the OVRTX renderer with internal environment cloning.

        Args:
            spec: Tiled camera description (resolution, paths, data types).
        """
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

        logger.info("Injecting camera definitions...")

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
            device_id=self._warp_device.ordinal,
            enable_shadows=self.cfg.enable_shadows,
        )
        self._render_product_paths.append(render_product_path)

        combined_usd_string = self._exported_usd_string + "\n\n" + render_product_string
        self._exported_usd_string = None  # Free memory

        # If temp_usd_dir is set, write the combined USD stage to a temporary file.
        if self.cfg.temp_usd_dir is not None:
            _write_file(Path(self.cfg.temp_usd_dir), "ovrtx_renderer_stage.usda", combined_usd_string)

        logger.info("Loading USD into OvRTX...")
        self._renderer.open_usd_from_string(combined_usd_string)
        logger.info("OVRTX loaded USD from string successfully")

        camera_paths = [f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(num_envs)]
        if num_envs > 1:
            self._clone_sources_in_ovrtx()
            self._update_scene_partitions_after_clone(num_envs)
            # OVRTX 0.4 keeps the initial Fabric camera relationship after clone_usd creates the remaining
            # cameras. Rewrite it so the RenderProduct includes every camera in its tiled output.
            self._renderer.write_array_attribute(
                prim_paths=[render_product_path],
                attribute_name="camera",
                tensors=[camera_paths],
            )

        self._initialized_scene = True

        self._camera_xform_binding = self._renderer.bind_attribute(
            prim_paths=camera_paths,
            attribute_name="omni:xform",
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.EXISTING_ONLY,
        )

        # OVRTX requires omni:resetXformStack on cameras for correct world transform binding
        self._renderer.write_attribute(
            prim_paths=camera_paths,
            attribute_name="omni:resetXformStack",
            tensor=np.full(num_envs, True, dtype=np.bool_),
        )

        if self._camera_xform_binding is not None:
            logger.info("Camera binding created successfully")
        else:
            raise RuntimeError("Camera binding is None — cannot render without a valid camera binding")

        self._setup_xform_bindings()
        self._setup_deformable_bindings(num_envs)
        self._setup_particle_bindings()
        self._setup_cable_bindings()

    def _clone_sources_in_ovrtx(self):
        """Clone sources in OVRTX using the scene :class:`~isaaclab.cloner.ClonePlan`."""
        clone_plan = self._clone_plan
        if clone_plan is None or clone_plan.env_ids is None or clone_plan.positions is None:
            raise RuntimeError("Clone plan with environment ids and positions is required when using OVRTX cloning")

        env_ids = clone_plan.env_ids
        clone_mask = clone_plan.clone_mask
        num_envs = len(env_ids)
        env_prim_paths = [f"/World/envs/env_{int(env_id)}" for env_id in env_ids]
        logger.info("Cloning sources in OVRTX...")

        num_cloned_sources = 0
        for row_idx, (source, destination) in enumerate(zip(clone_plan.sources, clone_plan.destinations, strict=True)):
            target_paths = [
                destination.format(int(env_id))
                for env_id in env_ids[clone_mask[row_idx]]
                if destination.format(int(env_id)) != source
            ]
            if target_paths:
                logger.debug("Cloning row %d: %s -> %d target(s)", row_idx, source, len(target_paths))
                try:
                    self._renderer.clone_usd(source, target_paths)
                    num_cloned_sources += 1
                except Exception as e:
                    error_msg = f"Failed to clone row {row_idx} from {source}: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

        logger.info("Cloned %d sources successfully in OVRTX", num_cloned_sources)
        env_root_xforms = np.tile(np.eye(4, dtype=np.float64), (num_envs, 1, 1))
        env_root_xforms[:, 3, :3] = clone_plan.positions
        self._renderer.write_attribute(
            prim_paths=env_prim_paths,
            attribute_name="omni:xform",
            tensor=env_root_xforms,
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.MUST_EXIST,
        )

    def _update_scene_partitions_after_clone(self, num_envs: int):
        """Update scene partition attributes on cloned environments and cameras in OvRTX."""
        logger.info("Writing scene partitions for %d environments...", num_envs)
        partition_tokens = [f"env_{i}" for i in range(num_envs)]
        env_prim_paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
        camera_prim_paths = [f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(num_envs)]

        self._renderer.write_attribute(
            env_prim_paths,
            "primvars:omni:scenePartition",
            partition_tokens,
            semantic=Semantic.TOKEN_STRING,
        )
        logger.info("Written primvars:omni:scenePartition to %d environments", num_envs)

        self._renderer.write_attribute(
            camera_prim_paths,
            "omni:scenePartition",
            partition_tokens,
            semantic=Semantic.TOKEN_STRING,
        )
        logger.info("Written omni:scenePartition to %d cameras", num_envs)

    def _setup_xform_bindings_legacy(self):
        """Setup OVRTX bindings for scene objects to sync with Newton physics."""
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            logger.debug("NewtonManager not available, skipping object bindings")
            return

        if SimulationContext.instance() is None:
            logger.info("No active simulation context, will not set up ovrtx object bindings for newton")
            return

        newton_model = NewtonManager.get_model()
        if newton_model is None:
            logger.debug("Newton model not available, skipping object bindings")
            return

        all_body_paths = getattr(newton_model, "body_label", None)
        if all_body_paths is None:
            logger.info("Newton model has no body_label, skipping object bindings")
            return

        object_paths = []
        newton_indices = []
        for idx, path in enumerate(all_body_paths):
            if "/World/envs/" in path and self._camera_rel_path not in path and "GroundPlane" not in path:
                object_paths.append(path)
                newton_indices.append(idx)

        if len(object_paths) == 0:
            logger.info("No dynamic objects found for binding")
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
        self._object_transform_buffer = wp.zeros(len(newton_indices), dtype=wp.mat44d, device=self._device)

    def _setup_deformable_bindings_legacy(self, num_envs: int):
        """Setup OVRTX bindings for Newton deformable bodies.

        Args:
            num_envs: Number of environments.
        """
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            logger.debug("NewtonManager not available, skipping deformable body bindings")
            return

        # Early return if the deformable registry is empty.
        deformable_registry = NewtonManager._deformable_registry
        if not deformable_registry:
            logger.debug("Deformable registry is empty, skipping deformable body bindings")
            return

        # Validate the number of particle offsets for each deformable entry upfront.
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

        vis_mesh_prim_paths: list[str] = []

        # Each registry entry is one deformable asset registered at spawn time. Its
        # ``vis_mesh_prim_path`` uses a regex env wildcard (e.g. ``env_.*``) to denote one
        # homogeneous visual mesh replicated into every environment, not a subset of envs.
        # During replication, Newton appends one particle block per env in contiguous env order
        # and records the start index in ``entry.particle_offsets``; ``particles_per_body`` is
        # the block size. The inner loop therefore emits one OVRTX mesh binding per env,
        # resolving the env wildcard with ``env_idx`` and pairing it with that env's slice in
        # the flat ``particle_q`` array.
        #
        # This mapping is valid only while deformable registry entries remain homogeneous across
        # all envs with dense, contiguous env ids. If deformables later support env subsets or
        # non-contiguous env ids, OVRTX must consume explicit per-instance env metadata instead
        # of deriving env ids from ``enumerate(entry.particle_offsets)``.
        for entry in deformable_registry:
            for idx, particle_offset in enumerate(entry.particle_offsets):
                self._deformable_particle_offsets.append(particle_offset)
                self._deformable_particle_counts.append(entry.particles_per_body)

                vis_mesh_prim_paths.append(
                    re.sub(r"(?<=[Ee]nv_)(?:\[\^/\][*+]|\.\*)", str(idx), entry.vis_mesh_prim_path)
                )

        prim_count = len(vis_mesh_prim_paths)
        if prim_count == 0:
            logger.warning("No deformable visual prim paths collected, skipping deformable body bindings")
            return

        # World-space particle_q is written directly into mesh points. Reset the xform stack
        # and pin identity omni:xform so inherited env/asset transforms are not applied twice.
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
        """Setup OVRTX ``points`` bindings for Newton cables (UsdGeom.BasisCurves).

        Cables are rigid segment bodies, not particles, so their curve points are derived from
        ``body_q`` each frame rather than sliced out of ``particle_q``.
        """
        discovered = self._discover_cable_segment_bindings()
        if discovered is None:
            return

        cable_prim_paths, flat_shape_ids, offsets, counts = discovered
        prim_count = len(cable_prim_paths)
        # Points are written in world space, so neutralise the inherited env/asset transform the
        # same way the deformable path does; otherwise the transform is applied twice.
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

        # Device-resident buffers select OVRTX's GPU-interop update path via DLPack device.
        self._allocate_cable_device_buffers(flat_shape_ids, offsets, counts)
        self._cable_point_slices = [
            self._cable_points[offset + curve : offset + curve + segment_count + 1]
            for curve, (offset, segment_count) in enumerate(zip(offsets, counts, strict=True))
        ]

    def _setup_particle_bindings_legacy(self) -> None:
        """Setup OVRTX bindings for Newton particle clouds."""
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            logger.debug("NewtonManager not available, skipping particle point bindings")
            return

        particle_visual_prims = NewtonManager._particle_visual_prims
        if not particle_visual_prims:
            logger.debug("No particle visual prims registered, skipping particle point bindings")
            return

        self._particle_visual_offsets = []
        self._particle_visual_counts = []
        points_prim_paths: list[str] = []

        for prim_path, record in particle_visual_prims.items():
            points_prim_paths.append(prim_path)
            self._particle_visual_offsets.append(record.offset)
            self._particle_visual_counts.append(record.count)

        prim_count = len(points_prim_paths)

        # World-space particle_q is written directly into points. Reset the xform stack
        # and pin identity omni:xform so inherited env/asset transforms are not applied twice.
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
        """Create OVRTX-specific RenderData with GPU buffers.

        Performs OVRTX initialization (stage export, USD load, bindings) on first call,
        matching the interface of Isaac RTX and Newton Warp which need no separate initialize().
        """
        # Resolve the device once through Warp: a bare "cuda" pins to Warp's current CUDA device
        # here, and the normalized string keeps every downstream consumer (kernel launches,
        # allocations) on that same device.
        self._warp_device = wp.get_device(spec.device)
        self._device = str(self._warp_device)
        if not self._initialized_scene:
            self._initialize_from_spec(spec)
        return OVRTXRenderData(spec, self._device)

    def set_outputs(self, render_data: OVRTXRenderData, output_data: dict[str, ProxyArray]) -> None:
        """Register pre-allocated warp output buffers for rendering.

        Each :class:`~isaaclab.utils.warp.ProxyArray` already carries the correct warp
        dtype from :meth:`~isaaclab.sensors.camera.CameraData.allocate`; store
        the underlying warp array directly. ``rgb`` is excluded because it is a
        non-contiguous strided view into ``rgba`` and is updated automatically.

        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`.
        """
        render_data.warp_buffers = {
            name: proxy.warp for name, proxy in output_data.items() if name != str(RenderBufferKind.RGB)
        }
        # When PPISP is composed but the user did not request the raw HDR AOV,
        # allocate an internal HDR scratch buffer under "rgb_hdr" so both the
        # HdrColor extractor and PPISP dispatch can use the same buffer map.
        if render_data.ppisp_pipeline is not None and str(RenderBufferKind.RGB_HDR) not in render_data.warp_buffers:
            ref_proxy = next(iter(output_data.values()))
            render_data.warp_buffers[str(RenderBufferKind.RGB_HDR)] = wp.zeros(
                (render_data.num_envs, render_data.height, render_data.width, 3),
                dtype=wp.float32,
                device=ref_proxy.device,
            )
        if render_data.ppisp_pipeline is not None:
            if str(RenderBufferKind.RGBA) not in render_data.warp_buffers:
                raise ValueError(
                    "OVRTX renderer ISP requires 'rgba' (or 'rgb', which aliases into rgba) as the"
                    " LDR output destination, but neither was provided. Add 'rgb' or 'rgba' to"
                    " Camera.cfg.data_types when isp_cfg is set."
                )

    def _update_transforms_legacy(self) -> None:
        """Sync transforms to OVRTX."""
        if (
            self._object_xform_binding is None
            or self._object_newton_indices is None
            or self._object_scales is None
            or self._object_transform_buffer is None
        ):
            return

        # If self._object_newton_indices is not None, then Newton's the current physics backend

        from isaaclab_newton.physics import NewtonManager

        newton_state = NewtonManager.get_state()
        if newton_state is None:
            raise RuntimeError("Newton state should not be None")

        body_q = getattr(newton_state, "body_q", None)
        if body_q is None:
            return

        wp.launch(
            kernel=sync_newton_transforms_kernel,
            dim=len(self._object_newton_indices),
            inputs=[self._object_transform_buffer, self._object_newton_indices, body_q, self._object_scales],
            device=self._device,
        )
        # Blocking ``write()`` so the buffer stays valid until OVRTX finishes reading it.
        # ``DataAccess.ASYNC`` + the Warp CUDA stream let OVRTX read in place and wait
        # on-GPU for the kernel; ``SYNC`` is rejected for GPU buffers.
        self._object_xform_binding.write(
            self._object_transform_buffer,
            data_access=DataAccess.ASYNC,
            cuda_stream=self._warp_device.stream.cuda_stream,
        )

    def _update_geometries_legacy(self) -> None:
        """Sync geometries to OVRTX."""
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
        """Recompute world-space cable curve points from Newton bodies and write them to OVRTX."""
        self._compute_cable_points_world()

        # Slices alias ``_cable_points``. Pass Warp's CUDA stream so OVRTX waits on-GPU instead of
        # forcing a host sync. ``DataAccess.ASYNC`` + device tensors select GPU interop; ``SYNC`` or
        # a host array silently takes the CPU path.
        self._cable_points_binding.write(
            cast(Any, self._cable_point_slices),
            data_access=DataAccess.ASYNC,
            cuda_stream=self._warp_device.stream.cuda_stream,
        )

    def _write_particle_q_slices(
        self,
        binding: Any,
        particle_offsets: list[int],
        particle_counts: list[int],
    ) -> None:
        """Write world-space ``particle_q`` slices into one OVRTX array-attribute binding.

        Args:
            binding: OVRTX array-attribute binding for the ``points`` attribute.
            particle_offsets: Start index of each prim's slice into Newton's ``particle_q``.
            particle_counts: Number of particles in each prim's slice.
        """
        from isaaclab_newton.physics import NewtonManager

        state = NewtonManager.get_state()
        if state is None:
            raise RuntimeError("Newton state should not be None")

        particle_q = getattr(state, "particle_q", None)
        if particle_q is None:
            raise RuntimeError("Newton state has no particle_q but particle geometry bindings exist")

        particle_slices = [
            particle_q[particle_offset : particle_offset + particle_count]
            for particle_offset, particle_count in zip(particle_offsets, particle_counts, strict=True)
        ]

        # Array attributes cannot use ``binding.map()`` like rigid-body xforms, and
        # ``DataAccess.ASYNC`` lets OVRTX read the slices in place (no copy on ingest).
        # Because the slices alias ``particle_q``, OVRTX must not read them until the
        # Warp kernels that wrote ``particle_q`` have finished. Passing ``cuda_stream``
        # hands OVRTX the Warp stream those kernels were enqueued on so it can insert a
        # GPU-side wait (a cross-stream dependency) before its read, instead of us
        # forcing a host-side ``wp.synchronize_device()`` that would stall the CPU.
        cuda_stream = self._warp_device.stream.cuda_stream
        binding.write(
            cast(Any, particle_slices),
            data_access=DataAccess.ASYNC,
            cuda_stream=cuda_stream,
        )

    def _update_camera_legacy(
        self,
        render_data: OVRTXRenderData,
        positions: ProxyArray,
        orientations: ProxyArray,
        intrinsics: ProxyArray,
    ) -> None:
        """Update camera transforms in OVRTX binding."""
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
            self._camera_xform_binding.write(
                camera_transforms,
                data_access=DataAccess.ASYNC,
                cuda_stream=self._warp_device.stream.cuda_stream,
            )

    def read_output(
        self,
        render_data: OVRTXRenderData,
        camera_data: CameraData,
    ) -> None:
        """Forward per-output metadata collected during :meth:`render` into ``camera_data.info``.

        This is a *replace*, not a *merge*: every seeded output key is reset to this frame's metadata,
        which is ``None`` when its render var was absent. Because :meth:`render` rebuilds ``renderer_info``
        from scratch each frame (see the ``renderer_info.clear()`` in :meth:`_process_render_frame`), a render
        var that disappears on a later frame (e.g. a missing ``SemanticIdMap``) must clear the corresponding
        ``camera_data.info`` entry too, or downstream consumers would keep reading stale labels. ``renderer_info``
        only ever holds a subset of the outputs, so iterating ``camera_data.info`` both preserves its
        ``output``-mirroring key set and resets any dropped metadata to ``None``.

        Present entries are stored by reference (a shallow assignment, not a deep copy): ``camera_data.info``
        shares the same metadata dict objects as ``render_data.renderer_info`` (e.g. the semantic
        ``idToLabels`` mapping). Those references stay valid even after ``renderer_info`` is cleared or
        rebuilt, and each render builds a fresh metadata dict, so no aliased object is mutated in place.

        Pixel data needs no handling here: :meth:`set_outputs` wraps each ``camera_data.output`` tensor as a
        zero-copy warp array stored in ``render_data.warp_buffers``, and :meth:`render` writes the rendered
        tiles directly into those warp arrays.

        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.read_output`.
        """
        assert camera_data.info is not None, "CameraData.info should be created in CameraData.allocate"
        for output_name in camera_data.info:
            camera_data.info[output_name] = render_data.renderer_info.get(output_name)

    def _generate_random_colors_from_ids(self, input_ids: wp.array, output_colors: wp.array | None) -> wp.array:
        """Generate pseudo-random RGBA colors from uint32 IDs into a reusable output buffer.

        Args:
            input_ids: 3-D uint32 Warp array of shape (H, W, 1).
            output_colors: Existing color buffer to reuse, or None to allocate a new one.

        Returns:
            Color buffer containing the generated colors.
        """

        # Lazily allocate, and re-allocate if the shape changes.
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
        """Map ``render_var`` for CUDA reads and yield it as a Warp array.

        The render is still in flight when the mapping returns, so reading it has to be ordered
        against render completion. Normally that is a ``cudaStreamWaitEvent`` on the Warp stream the
        consuming kernels run on, which is the ordering the OVRTX API is designed around.

        On Linux that GPU-side wait measures substantially slower end to end, so the mapping is
        instead requested with no GPU-side barrier and the calling thread blocks on the
        render-completion event. Setting :data:`_DISABLE_LINUX_CUDA_CPU_SYNC_ENV` to ``1`` puts
        Linux back on the GPU-side wait; it is an escape hatch for platforms where that trade-off
        no longer holds, and is worth re-measuring before being relied on.

        Note that ``sync_stream=0`` is OVRTX's "no sync" sentinel, *not* the NULL CUDA stream: the
        field encodes ``0=no sync, 1=default stream, >1=specific stream``, so omitting the argument
        entirely means ``1``, not ``0``.

        The yielded array is a zero-copy view of the mapped memory and is only valid inside the
        ``with`` block -- the mapping is released on exit.

        Args:
            render_var: OVRTX ``RenderVarOutput`` to map (looked up from ``frame.render_vars``).

        Yields:
            The render var's contents as a Warp array, valid for the duration of the context.
        """
        gpu_side_sync = _gpu_side_render_var_sync_enabled()
        sync_stream = self._warp_device.stream.cuda_stream if gpu_side_sync else 0
        with render_var.map(device=Device.CUDA, sync_stream=sync_stream) as mapping:
            if not gpu_side_sync:
                mapping.wait()
            yield wp.from_dlpack(mapping)

    def _process_id_segmentation_render_var(
        self,
        render_data: OVRTXRenderData,
        frame,
        output_buffers: dict,
        render_var_key: str,
        buffer_key: str,
        colorize: bool,
    ) -> None:
        """Extract a uint32 ID-segmentation render var into ``output_buffers[buffer_key]``.

        Shared by ``semantic_segmentation`` (``SemanticSegmentation``) and ``instance_segmentation``
        (``NonStableInstanceSegmentation``), which only differ in the source render var, the destination buffer,
        and whether to colorize.

        Args:
            render_data: OVRTX render data for the current frame.
            frame: OVRTX frame holding the mapped render vars.
            output_buffers: Destination warp buffers, keyed by data type.
            render_var_key: ``frame.render_vars`` key of the OVRTX render var to read.
            buffer_key: Data type key into ``output_buffers``.
            colorize: If True, IDs are mapped to RGBA colors; otherwise raw uint32 IDs are copied.
        """
        render_var = frame.render_vars.get(render_var_key)
        if render_var is None or buffer_key not in output_buffers:
            return

        with self._map_render_var_to_dlpack(render_var) as tiled_data:
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
                # Non-colorized: ensure (TH, TW, 1) shape for the uint32 extraction kernel. Reshape the warp
                # array directly instead of round-tripping through torch, which raises on ``torch.uint32``
                # (newer torch exposes the dtype but ``wp.from_torch`` still rejects it).
                if tiled_data.ndim == 2:
                    tiled_data = tiled_data.reshape((*tiled_data.shape, 1))
                self._launch_extract_all_tiles(render_data, tiled_data, output_buffers[buffer_key])

    def _process_semantic_id_map(self, render_data: OVRTXRenderData, frame) -> None:
        """Decode the ``SemanticIdMap`` render var into ``render_data.renderer_info["semantic_segmentation"]``.

        Populates an ``"idToLabels"`` mapping compatible with Isaac RTX / Replicator: keys are the raw semantic
        IDs (``colorize_semantic_segmentation=False``) or the RGBA color tuples the segmentation buffer uses
        (``colorize_semantic_segmentation=True``); values are ``{semantic_type: label}`` dicts. The reserved
        BACKGROUND (ID 0) and UNLABELLED (ID 1) entries are always included.

        Args:
            render_data: OVRTX render data for the current frame.
            frame: OVRTX frame holding the mapped render vars.
        """
        semantic_id_map = frame.render_vars.get(_SEMANTIC_ID_MAP_VAR)
        if semantic_id_map is None:
            return

        with semantic_id_map.map(device=Device.CPU) as mapping:
            labels_by_id = decode_semantic_id_map(np.from_dlpack(mapping))

        render_data.renderer_info["semantic_segmentation"] = {
            "idToLabels": build_semantic_id_to_labels(
                labels_by_id, colorize=self.cfg.colorize_semantic_segmentation, device=self._device
            )
        }

    def _process_instance_segmentation_maps(self, render_data: OVRTXRenderData, frame) -> None:
        """Decode the instance-segmentation map render vars into ``renderer_info["instance_segmentation"]``.

        An *instance pixel ID* is a compact integer that the renderer assigns to each visible object instance.
        Every pixel in the segmentation buffer holds the ID of the instance rendered at that location; the same
        ID maps to the same object across the entire frame.  ID 0 is reserved for BACKGROUND (no geometry), and
        ID 1 for UNLABELLED (geometry with no semantic annotation).  All other IDs are dynamically assigned per
        frame.

        Populates ``"idToLabels"`` (instance pixel ID -> USD prim path) and ``"idToSemantics"`` (instance pixel
        ID -> ``{semantic_type: label}``) compatible with Isaac RTX / Replicator. Resolving both requires all
        three map render vars — ``StableIdSemanticIdMap`` (pixel ID -> stable ID + semantic ID), ``StableIdMap``
        (stable ID -> prim path), and ``SemanticIdMap`` (semantic ID -> label). Keys are the raw pixel IDs
        (``colorize_instance_segmentation=False``) or the RGBA color tuples the segmentation buffer uses
        (``colorize_instance_segmentation=True``); the reserved BACKGROUND (ID 0) and UNLABELLED (ID 1) entries
        are always included.

        Raises:
            RuntimeError: If any of the three required render vars is absent from ``frame``.

        Args:
            render_data: OVRTX render data for the current frame.
            frame: OVRTX frame holding the mapped render vars.
        """
        resolved = {key: frame.render_vars.get(key) for key in _INSTANCE_SEGMENTATION_MAP_VARS}
        missing = [key for key, render_var in resolved.items() if render_var is None]
        if missing:
            raise RuntimeError(
                f"instance_segmentation was requested but the following render vars are missing from the "
                f"OVRTX frame: {missing}. Available vars: {list(frame.render_vars.keys())}"
            )

        with resolved[_STABLE_ID_SEMANTIC_ID_MAP_VAR].map(device=Device.CPU) as mapping:
            stable_id_semantic_id_map = decode_stable_id_semantic_id_map(np.from_dlpack(mapping))
        with resolved[_STABLE_ID_MAP_VAR].map(device=Device.CPU) as mapping:
            stable_id_to_path = decode_stable_id_map(np.from_dlpack(mapping))
        with resolved[_SEMANTIC_ID_MAP_VAR].map(device=Device.CPU) as mapping:
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
        """Launch ``extract_all_tiles_kernel`` for one tiled/output buffer pair.

        This is the only place that should launch ``extract_all_tiles_kernel``: it validates that
        ``output_buffer`` cannot read past the end of ``tiled_buffer`` (the kernel derives its per-thread
        channel loop bound from ``output_buffer``'s last dimension) before every launch, so callers cannot
        accidentally skip the check.

        Args:
            render_data: OVRTX render data for the current frame.
            tiled_buffer: 3D array of shape (H, W, C) holding all tiles packed into one buffer.
            output_buffer: 4D array of shape (num_envs, H, W, C) to receive the per-env tiles, with C no
                greater than ``tiled_buffer``'s channel count.

        Raises:
            ValueError: If ``output_buffer``'s channel count exceeds ``tiled_buffer``'s.
        """
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
            inputs=[
                tiled_buffer,
                output_buffer,
                render_data.num_cols,
                render_data.width,
                render_data.height,
            ],
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
        """Extract per-env RGBA tiles from tiled buffer into output_buffers (single kernel launch)."""
        output_buffer = output_buffers[buffer_key]
        num_channels = output_buffer.shape[-1]
        if num_channels not in (3, 4):
            raise ValueError(f"Expected RGB (3 channels) or RGBA (4 channels), got {num_channels}")

        self._launch_extract_all_tiles(render_data, tiled_data, output_buffer)

    def _extract_depth_tiles(
        self,
        render_data: OVRTXRenderData,
        tiled_depth_data: wp.array,
        output_buffers: dict,
        buffer_keys: Sequence[str],
    ) -> None:
        """Extract per-env depth tiles into the given output buffers (one kernel launch each).

        Args:
            render_data: OVRTX render data for the current frame.
            tiled_depth_data: Tiled depth data mapped from one depth render var.
            output_buffers: Destination warp buffers, keyed by data type.
            buffer_keys: Data types that this depth render var measures. Keys absent from
                ``output_buffers`` are skipped.
        """
        for depth_type in buffer_keys:
            if depth_type in output_buffers:
                self._launch_extract_all_tiles(render_data, tiled_depth_data, output_buffers[depth_type])

    def _extract_hdr_color_tiles(
        self, render_data: OVRTXRenderData, tiled_data: wp.array, output_buffers: dict
    ) -> None:
        """Extract per-env HdrColor tiles into output_buffers."""
        if "rgb_hdr" not in output_buffers:
            return
        if tiled_data.dtype not in (wp.float16, wp.float32):
            raise TypeError(f"Unsupported OVRTX HdrColor dtype: {tiled_data.dtype}.")
        self._launch_extract_all_tiles(render_data, tiled_data, output_buffers["rgb_hdr"])

    def _prepare_ppisp_hdr_source(
        self, render_data: OVRTXRenderData, tiled_data: wp.array, output_buffers: dict
    ) -> wp.array:
        """Return the PPISP HdrColor source on the output buffer device."""
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
        """Extract RGB, depth, albedo, and semantic from a single render frame into output_buffers."""
        # Reset per-output metadata so it is a snapshot of this frame only. Unlike pixel AOVs (always
        # present), metadata like the semantic ``idToLabels`` is only repopulated below when its render var
        # is available, so without this a missing SemanticIdMap on a later frame would leave a stale mapping.
        render_data.renderer_info.clear()

        ldr_color = frame.render_vars.get(_LDR_COLOR_VAR)
        if ldr_color is not None:
            buffer_key = None

            if render_data.ppisp_pipeline is None and "rgba" in output_buffers:
                buffer_key = "rgba"
            else:
                # The output buffers must contain only one simple shading data type at most after resolution of the data
                # types during creation of the output buffers (OVRTXRenderData._create_warp_buffers).
                for dt in _RTX_MINIMAL_MODES:
                    if dt in output_buffers:
                        buffer_key = dt
                        break

            if buffer_key is not None:
                with self._map_render_var_to_dlpack(ldr_color) as tiled_data:
                    self._extract_rgba_tiles(render_data, tiled_data, output_buffers, buffer_key)

        for depth_var, buffer_keys in _DEPTH_VAR_BUFFER_KEYS.items():
            depth_render_var = frame.render_vars.get(depth_var)
            if depth_render_var is None:
                continue
            if not any(buffer_key in output_buffers for buffer_key in buffer_keys):
                continue
            with self._map_render_var_to_dlpack(depth_render_var) as tiled_depth_data:
                if tiled_depth_data.dtype == wp.uint32:
                    tiled_depth_data = wp.from_torch(
                        wp.to_torch(tiled_depth_data).view(torch.float32), dtype=wp.float32
                    )
                self._extract_depth_tiles(render_data, tiled_depth_data, output_buffers, buffer_keys)

        albedo_var = frame.render_vars.get(_ALBEDO_VAR)
        if albedo_var is not None and "albedo" in output_buffers:
            with self._map_render_var_to_dlpack(albedo_var) as tiled_albedo_data:
                self._extract_rgba_tiles(render_data, tiled_albedo_data, output_buffers, "albedo", suffix="albedo")

        hdr_color = frame.render_vars.get(_HDR_COLOR_VAR)
        if hdr_color is not None and "rgb_hdr" in output_buffers:
            with self._map_render_var_to_dlpack(hdr_color) as tiled_hdr_data:
                tiled_hdr_data = self._prepare_ppisp_hdr_source(render_data, tiled_hdr_data, output_buffers)
                self._extract_hdr_color_tiles(render_data, tiled_hdr_data, output_buffers)

        self._process_id_segmentation_render_var(
            render_data,
            frame,
            output_buffers,
            _SEMANTIC_SEGMENTATION_VAR,
            "semantic_segmentation",
            self.cfg.colorize_semantic_segmentation,
        )
        # Decode the SemanticIdMap into camera.data.info["semantic_segmentation"]["idToLabels"].
        if "semantic_segmentation" in output_buffers:
            self._process_semantic_id_map(render_data, frame)

        self._process_id_segmentation_render_var(
            render_data,
            frame,
            output_buffers,
            _INSTANCE_SEGMENTATION_VAR,
            "instance_segmentation",
            self.cfg.colorize_instance_segmentation,
        )
        # Decode the StableIdSemanticIdMap/StableIdMap/SemanticIdMap trio into
        # camera.data.info["instance_segmentation"]["idToLabels"] and ["idToSemantics"].
        if "instance_segmentation" in output_buffers:
            self._process_instance_segmentation_maps(render_data, frame)

        normals_var = frame.render_vars.get(_NORMALS_VAR)
        if normals_var is not None and "normals" in output_buffers:
            with self._map_render_var_to_dlpack(normals_var) as tiled_normals_data:
                self._launch_extract_all_tiles(render_data, tiled_normals_data, output_buffers["normals"])

        # For motion vectors, extract only the first two (u, v) channels from the tiled buffer.
        # Note: mirrors the Isaac RTX renderer's handling of the "TargetMotionSD" AOV
        # (check: https://github.com/isaac-sim/IsaacLab/issues/2003).
        motion_var = frame.render_vars.get(_MOTION_VECTORS_VAR)
        if motion_var is not None and "motion_vectors" in output_buffers:
            with self._map_render_var_to_dlpack(motion_var) as tiled_motion_vectors_data:
                self._launch_extract_all_tiles(render_data, tiled_motion_vectors_data, output_buffers["motion_vectors"])

    def _render_legacy(self, render_data: OVRTXRenderData) -> None:
        """Render the scene into the provided RenderData."""
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call initialize() first.")
        if self._renderer is None or len(self._render_product_paths) == 0:
            return
        material_writer = self._visual_material_writer_ref() if self._visual_material_writer_ref is not None else None
        try:
            if material_writer is not None:
                material_writer.publish()
            products = self._renderer.step(
                render_products=set(self._render_product_paths),
                delta_time=1.0 / 60.0,
            )
        finally:
            if material_writer is not None:
                drain_errors = contextlib.nullcontext() if sys.exc_info()[0] is None else contextlib.suppress(Exception)
                with drain_errors:
                    material_writer.drain()
        product_path = self._render_product_paths[0]
        if product_path in products and len(products[product_path].frames) > 0:
            self._process_render_frame(
                render_data,
                products[product_path].frames[0],
                render_data.warp_buffers,
            )

        # Post-render PPISP: HDR scene-linear → LDR RGBA. Source/destination
        # buffers are the same warp buffer map used by extraction.
        if render_data.ppisp_pipeline is not None:
            render_data.ppisp_pipeline.apply(
                render_data.warp_buffers[str(RenderBufferKind.RGB_HDR)],
                render_data.warp_buffers[str(RenderBufferKind.RGBA)],
            )

    def _close_legacy(self) -> None:
        """Release the renderer's tensor bindings and stage. See :meth:`close`."""

        # Unbind before tearing down renderer
        def _safe_unbind(binding, name: str) -> None:
            if binding is None:
                return
            try:
                binding.unbind()
            except Exception as e:
                if "destroyed" not in str(e).lower():
                    logger.warning("Error unbinding %s: %s", name, e)

        _safe_unbind(self._camera_xform_binding, "camera transforms")
        self._camera_xform_binding = None
        _safe_unbind(self._object_xform_binding, "object transforms")
        self._object_xform_binding = None
        self._object_transform_buffer = None
        _safe_unbind(self._deformable_points_binding, "deformable points")
        self._deformable_points_binding = None
        _safe_unbind(self._particle_points_binding, "particle points")
        self._particle_points_binding = None
        _safe_unbind(self._cable_points_binding, "cable points")
        self._cable_points_binding = None

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
            except Exception as e:
                logger.warning("Error resetting stage: %s", e)

            self._renderer = None

        self._render_product_paths.clear()
        self._output_id_color_buffers.clear()
        self._initialized_scene = False

    # ---------------------------------------------------------------------------
    # Dispatch methods — route to ovstage or legacy implementation
    # ---------------------------------------------------------------------------

    def _init_fields(self) -> None:
        if self._use_ovstage:
            self._init_fields_ovstage()
        else:
            self._init_fields_legacy()

    def _initialize_from_spec(self, spec: CameraRenderSpec) -> None:
        if self._use_ovstage:
            self._initialize_from_spec_ovstage(spec)
        else:
            self._initialize_from_spec_legacy(spec)

    def _setup_xform_bindings(self) -> None:
        if self._use_ovstage:
            self._setup_xform_bindings_ovstage()
        else:
            self._setup_xform_bindings_legacy()

    def _setup_deformable_bindings(self, num_envs: int) -> None:
        if self._use_ovstage:
            self._setup_deformable_bindings_ovstage(num_envs)
        else:
            self._setup_deformable_bindings_legacy(num_envs)

    def _setup_particle_bindings(self) -> None:
        if self._use_ovstage:
            self._setup_particle_bindings_ovstage()
        else:
            self._setup_particle_bindings_legacy()

    def _setup_cable_bindings(self) -> None:
        if self._use_ovstage:
            self._setup_cable_bindings_ovstage()
        else:
            self._setup_cable_bindings_legacy()

    @staticmethod
    def _discover_cable_segment_bindings() -> tuple[list[str], list[int], list[int], list[int]] | None:
        """Collect cable prim paths and packed Newton segment shape ids, or ``None`` to skip.

        Returns:
            ``None`` when Newton is unavailable or no renderable cables exist. Otherwise a tuple
            ``(cable_prim_paths, flat_shape_ids, offsets, counts)`` where:

            * ``cable_prim_paths``: concrete ``BasisCurves`` prim paths to bind, one per curve.
            * ``flat_shape_ids``: Newton shape ids for all curves, packed contiguously in segment
              order (indices into ``model.shape_body`` / ``shape_transform`` / ``shape_scale``).
            * ``offsets``: start index into ``flat_shape_ids`` for each curve.
            * ``counts``: number of segment shapes (capsules) for each curve.
        """
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            logger.debug("NewtonManager not available, skipping cable point bindings")
            return None

        cable_segment_shape_ids = NewtonManager.collect_cable_segment_shape_ids()
        if not cable_segment_shape_ids:
            logger.debug("No renderable Newton cables found, skipping cable point bindings")
            return None

        cable_prim_paths: list[str] = []
        flat_shape_ids: list[int] = []
        offsets: list[int] = []
        counts: list[int] = []
        for prim_path, segment_shape_ids in cable_segment_shape_ids.items():
            cable_prim_paths.append(prim_path)
            offsets.append(len(flat_shape_ids))
            counts.append(len(segment_shape_ids))
            flat_shape_ids.extend(segment_shape_ids)
        return cable_prim_paths, flat_shape_ids, offsets, counts

    def _allocate_cable_device_buffers(self, flat_shape_ids: list[int], offsets: list[int], counts: list[int]) -> None:
        """Allocate shared Warp arrays used by the cable point kernel."""
        device = self._device
        self._cable_shape_ids = wp.array(flat_shape_ids, dtype=wp.int32, device=device)
        self._cable_offsets = wp.array(offsets, dtype=wp.int32, device=device)
        self._cable_counts = wp.array(counts, dtype=wp.int32, device=device)
        self._cable_segment_counts = counts
        self._cable_max_points = max(counts) + 1 if counts else 0
        self._cable_points = wp.zeros(sum(count + 1 for count in counts), dtype=wp.vec3f, device=device)

    def _compute_cable_points_world(self) -> None:
        """Launch the cable endpoint kernel into ``_cable_points``."""
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
        """Sync transforms to OVRTX."""
        if self._use_ovstage:
            self._update_transforms_ovstage()
        else:
            self._update_transforms_legacy()

    def update_geometries(self) -> None:
        """Sync geometries to OVRTX."""
        if self._use_ovstage:
            self._update_geometries_ovstage()
        else:
            self._update_geometries_legacy()

    def update_camera(
        self,
        render_data: OVRTXRenderData,
        positions: ProxyArray,
        orientations: ProxyArray,
        intrinsics: ProxyArray,
    ) -> None:
        """Update camera transforms in OVRTX."""
        if self._use_ovstage:
            self._update_camera_ovstage(render_data, positions, orientations, intrinsics)
        else:
            self._update_camera_legacy(render_data, positions, orientations, intrinsics)

    def render(self, render_data: OVRTXRenderData) -> None:
        """Render the scene into the provided RenderData."""
        if self._use_ovstage:
            self._render_ovstage(render_data)
        else:
            self._render_legacy(render_data)

    def cleanup(self, render_data: OVRTXRenderData | None) -> None:
        """Release the render data's buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`.

        The stage queries, tensor bindings and render products this renderer holds are shared by
        every camera that resolves to it, so releasing them here would tear the scene down while
        the other cameras are still rendering. :meth:`close` releases them instead.
        """
        if render_data is None:
            return
        render_data.warp_buffers.clear()
        render_data.renderer_info.clear()
        render_data.ppisp_pipeline = None

    def close(self) -> None:
        """Release the shared stage state. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.close`."""
        if self._use_ovstage:
            self._close_ovstage()
        else:
            self._close_legacy()
        self._visual_material_writer_ref = None

    # ---------------------------------------------------------------------------
    # ovstage implementation
    #
    # Follow-up:
    # - Experiment with dropping the per-frame ``.wait()`` calls. ``advance_write_floor(N).wait()`` in
    #   :meth:`_render_ovstage` already bars all writes at ordinals <= N, so accumulate the
    #   ``Operation`` objects and ``stage.release_op(op.op_id)`` after it. They must outlive the
    #   barrier — an ``Operation`` is its buffer's only keepalive. Saves caller-side blocking only.
    # ---------------------------------------------------------------------------

    def _init_fields_ovstage(self) -> None:
        self._stage = None
        self._stage_paths = None
        self._ovstage_exit_stack: contextlib.ExitStack | None = None
        self._current_ordinal: int = 0
        self._camera_xform_query = None
        self._camera_paths_list = None
        self._object_xform_query = None
        self._object_paths_list = None
        self._deformable_points_query = None
        self._deformable_paths_list = None
        self._particle_points_query = None
        self._particle_paths_list = None
        self._cable_points_query = None
        self._cable_paths_list = None
        # DLTensor descriptors aliasing ``_cable_point_slices``; rebuilt only when cables rebind.
        self._cable_point_tensors: list = []

    def _initialize_from_spec_ovstage(self, spec: CameraRenderSpec) -> None:
        """Initialize the OVRTX renderer with internal environment cloning (ovstage path).

        Args:
            spec: Tiled camera description (resolution, paths, data types).
        """
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

        logger.info("Injecting camera definitions...")

        if self._exported_usd_string is None:
            raise RuntimeError("Expected an exported USD string from stage")

        render_product_string, render_product_path = build_render_product_as_string(
            width=width,
            height=height,
            num_envs=num_envs,
            data_types=data_types,
            minimal_mode=_resolve_rtx_minimal_mode(data_types),
            camera_rel_path=self._camera_rel_path,
            device_id=self._warp_device.ordinal,
            enable_shadows=self.cfg.enable_shadows,
        )
        self._render_product_paths.append(render_product_path)

        combined_usd_string = self._exported_usd_string + "\n\n" + render_product_string
        self._exported_usd_string = None  # Free memory

        # If temp_usd_dir is set, write the combined USD stage to a temporary file.
        if self.cfg.temp_usd_dir is not None:
            _write_file(Path(self.cfg.temp_usd_dir), "ovrtx_renderer_stage.usda", combined_usd_string)

        logger.info("Loading USD into OvRTX via ovstage...")
        self._ovstage_exit_stack = contextlib.ExitStack()
        self._stage = self._ovstage_exit_stack.enter_context(create_ovstage("isaaclab.ovrtx"))
        self._stage_paths = self._ovstage_exit_stack.enter_context(ovstage.PathDictionary(self._stage))
        # Ordinal 0 is the empty/unwritten state in ovstage; the first write must use >= 1.
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

        # Re-author the RenderProduct's camera relationship after clone. ``stage.clone`` recreates the per-env
        # cameras, so the RenderProduct must be pointed at the freshly-interned camera path ids to discover every
        # camera for tiled rendering.
        render_product_paths = self._stage_paths.create_path_list_from_strings([render_product_path])
        with self._stage.query_from_path_list(render_product_paths) as render_product_query:
            camera_attribute = self._stage_paths.intern_token("camera")
            camera_target_ids = np.array(
                [self._stage_paths.intern_path(path) for path in camera_paths], dtype=np.uint64
            )
            self._stage.write_attribute(
                render_product_query,
                camera_attribute,
                ordinal=self._current_ordinal,
                tensors=camera_target_ids,
                is_array=True,
                semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
            ).wait()
        self._stage_paths.destroy_path_list(render_product_paths)

        self._camera_paths_list = self._stage_paths.create_path_list_from_strings(camera_paths)
        self._camera_xform_query = self._stage.query_from_path_list(self._camera_paths_list)

        if self._camera_xform_query is None:
            raise RuntimeError("Camera query is None — cannot render without a valid camera query")
        logger.info("Camera query created successfully")

        # Resetting the xform stack makes omni:xform the absolute world transform, preventing
        # ancestor transforms (env root, asset root) from compounding on top of the camera pose.
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

        # Commit all init-time writes then attach. attach_ovstage happens last so the renderer
        # immediately sees the fully-configured scene on its first step.
        self._stage.advance_write_floor(ordinal=self._current_ordinal).wait()
        self._renderer.attach_ovstage(self._stage)
        logger.info("OVRTX loaded USD from string successfully via ovstage")
        self._current_ordinal += 1

    def _clone_sources_ovstage(self):
        """Clone sources in OVRTX using the scene :class:`~isaaclab.cloner.ClonePlan` (ovstage path)."""
        clone_plan = self._clone_plan
        if clone_plan is None or clone_plan.env_ids is None or clone_plan.positions is None:
            raise RuntimeError("Clone plan with environment ids and positions is required when using OVRTX cloning")

        env_ids = clone_plan.env_ids
        clone_mask = clone_plan.clone_mask
        num_envs = len(env_ids)
        env_prim_paths = [f"/World/envs/env_{int(env_id)}" for env_id in env_ids]

        logger.info("Cloning sources in OVRTX...")

        num_cloned_sources = 0
        for row_idx, (source, destination) in enumerate(zip(clone_plan.sources, clone_plan.destinations, strict=True)):
            target_paths = [
                destination.format(int(env_id))
                for env_id in env_ids[clone_mask[row_idx]]
                if destination.format(int(env_id)) != source
            ]
            if target_paths:
                logger.debug("Cloning row %d: %s -> %d target(s)", row_idx, source, len(target_paths))
                try:
                    self._stage.clone(source, target_paths, ordinal=self._current_ordinal)
                    num_cloned_sources += 1
                except Exception as e:
                    error_msg = f"Failed to clone row {row_idx} from {source}: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

        logger.info("Cloned %d sources successfully in OVRTX", num_cloned_sources)
        env_root_xforms = np.tile(np.eye(4, dtype=np.float64), (num_envs, 1, 1))
        env_root_xforms[:, 3, :3] = clone_plan.positions
        env_paths_list = self._stage_paths.create_path_list_from_strings(env_prim_paths)
        env_query = self._stage.query_from_path_list(env_paths_list)
        self._stage.write_attribute(
            env_query,
            "omni:xform",
            ordinal=self._current_ordinal,
            tensors=xform_tensor_from_numpy(env_root_xforms),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
        ).wait()

        self._stage.release_query(env_query).wait()
        self._stage_paths.destroy_path_list(env_paths_list)

    def _update_scene_partitions_after_clone_ovstage(self, num_envs: int):
        """Update scene partition attributes on cloned environments and cameras (ovstage path)."""
        logger.info("Writing scene partitions for %d environments...", num_envs)
        env_prim_paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
        camera_prim_paths = [f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(num_envs)]
        # TOKEN_ID semantic tells ovstage the uint64 values are interned string tokens, not raw integers;
        # the renderer resolves them back to the original "env_N" strings for scene-partition lookup.
        token_ids = np.array([self._stage_paths.intern_token(f"env_{i}") for i in range(num_envs)], dtype=np.uint64)

        env_paths_list = self._stage_paths.create_path_list_from_strings(env_prim_paths)
        env_query = self._stage.query_from_path_list(env_paths_list)
        self._stage.write_attribute(
            env_query,
            "primvars:omni:scenePartition",
            ordinal=self._current_ordinal,
            tensors=token_ids,
            is_array=False,
            semantic=ovstage.AttributeSemantic.TOKEN_ID,
        ).wait()
        self._stage.release_query(env_query).wait()
        self._stage_paths.destroy_path_list(env_paths_list)
        logger.info("Written primvars:omni:scenePartition to %d environments", num_envs)

        cam_paths_list = self._stage_paths.create_path_list_from_strings(camera_prim_paths)
        cam_query = self._stage.query_from_path_list(cam_paths_list)
        self._stage.write_attribute(
            cam_query,
            "omni:scenePartition",
            ordinal=self._current_ordinal,
            tensors=token_ids,
            is_array=False,
            semantic=ovstage.AttributeSemantic.TOKEN_ID,
        ).wait()
        self._stage.release_query(cam_query).wait()
        self._stage_paths.destroy_path_list(cam_paths_list)
        logger.info("Written omni:scenePartition to %d cameras", num_envs)

    def _setup_xform_bindings_ovstage(self) -> None:
        """Setup OVRTX bindings for scene objects to sync with Newton physics (ovstage path)."""
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            logger.debug("NewtonManager not available, skipping object bindings")
            return

        if SimulationContext.instance() is None:
            logger.info("No active simulation context, will not set up ovrtx object bindings for newton")
            return

        newton_model = NewtonManager.get_model()
        if newton_model is None:
            logger.debug("Newton model not available, skipping object bindings")
            return

        all_body_paths = getattr(newton_model, "body_label", None)
        if all_body_paths is None:
            logger.info("Newton model has no body_label, skipping object bindings")
            return

        object_paths = []
        newton_indices = []
        for idx, path in enumerate(all_body_paths):
            if "/World/envs/" in path and self._camera_rel_path not in path and "GroundPlane" not in path:
                object_paths.append(path)
                newton_indices.append(idx)

        if len(object_paths) == 0:
            logger.info("No dynamic objects found for binding")
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

        if self._object_xform_query is None:
            raise RuntimeError("Failed to create OVRTX object bindings")

        self._object_newton_indices = wp.array(newton_indices, dtype=wp.int32, device=self._device)
        self._object_scales = self._create_object_scale_array(object_paths)

    def _setup_deformable_bindings_ovstage(self, num_envs: int) -> None:
        """Setup OVRTX bindings for Newton deformable bodies (ovstage path).

        Args:
            num_envs: Number of environments.
        """
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            logger.debug("NewtonManager not available, skipping deformable body bindings")
            return

        # Early return if the deformable registry is empty.
        deformable_registry = NewtonManager._deformable_registry
        if not deformable_registry:
            logger.debug("Deformable registry is empty, skipping deformable body bindings")
            return

        # Validate the number of particle offsets for each deformable entry upfront.
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

        vis_mesh_prim_paths: list[str] = []

        # Each registry entry is one deformable asset registered at spawn time. Its
        # ``vis_mesh_prim_path`` uses a regex env wildcard (e.g. ``env_.*``) to denote one
        # homogeneous visual mesh replicated into every environment, not a subset of envs.
        # During replication, Newton appends one particle block per env in contiguous env order
        # and records the start index in ``entry.particle_offsets``; ``particles_per_body`` is
        # the block size. The inner loop therefore emits one OVRTX mesh binding per env,
        # resolving the env wildcard with ``env_idx`` and pairing it with that env's slice in
        # the flat ``particle_q`` array.
        #
        # This mapping is valid only while deformable registry entries remain homogeneous across
        # all envs with dense, contiguous env ids. If deformables later support env subsets or
        # non-contiguous env ids, OVRTX must consume explicit per-instance env metadata instead
        # of deriving env ids from ``enumerate(entry.particle_offsets)``.
        for entry in deformable_registry:
            for idx, particle_offset in enumerate(entry.particle_offsets):
                self._deformable_particle_offsets.append(particle_offset)
                self._deformable_particle_counts.append(entry.particles_per_body)

                vis_mesh_prim_paths.append(
                    re.sub(r"(?<=[Ee]nv_)(?:\[\^/\][*+]|\.\*)", str(idx), entry.vis_mesh_prim_path)
                )

        prim_count = len(vis_mesh_prim_paths)
        if prim_count == 0:
            logger.warning("No deformable visual prim paths collected, skipping deformable body bindings")
            return

        self._deformable_paths_list = self._stage_paths.create_path_list_from_strings(vis_mesh_prim_paths)
        self._deformable_points_query = self._stage.query_from_path_list(self._deformable_paths_list)

        # particle_q is already in world space, so resetting the xform stack and pinning an identity
        # omni:xform prevents the env-root and asset-root ancestor transforms from being applied on top.
        self._stage.write_attribute(
            self._deformable_points_query,
            "omni:resetXformStack",
            ordinal=self._current_ordinal,
            tensors=np.full(prim_count, True, dtype=np.bool_),
            is_array=False,
        ).wait()

        identity_xforms = np.tile(np.eye(4, dtype=np.float64), (prim_count, 1, 1))
        self._stage.write_attribute(
            self._deformable_points_query,
            "omni:xform",
            ordinal=self._current_ordinal,
            tensors=xform_tensor_from_numpy(identity_xforms),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
        ).wait()

        if self._deformable_points_query is None:
            raise RuntimeError("Failed to create OVRTX deformable body bindings")

    def _setup_cable_bindings_ovstage(self) -> None:
        """Setup ovstage ``points`` bindings for Newton cables (``UsdGeom.BasisCurves``).

        Mirrors :meth:`_setup_cable_bindings_legacy`: the endpoint kernel writes device memory and
        the per-frame handover is zero-copy. The per-curve slices and their DLTensor descriptors are
        built once here rather than per frame, because the layout is fixed for the lifetime of the
        binding — only the contents of ``_cable_points`` change each step.
        """
        discovered = self._discover_cable_segment_bindings()
        if discovered is None:
            return

        cable_prim_paths, flat_shape_ids, offsets, counts = discovered
        prim_count = len(cable_prim_paths)
        self._cable_paths_list = self._stage_paths.create_path_list_from_strings(cable_prim_paths)
        self._cable_points_query = self._stage.query_from_path_list(self._cable_paths_list)

        # The kernel emits world space, so reset the xform stack and pin an identity omni:xform to
        # stop the env-root and asset-root ancestor transforms being applied on top.
        self._stage.write_attribute(
            self._cable_points_query,
            "omni:resetXformStack",
            ordinal=self._current_ordinal,
            tensors=np.full(prim_count, True, dtype=np.bool_),
            is_array=False,
        ).wait()

        identity_xforms = np.tile(np.eye(4, dtype=np.float64), (prim_count, 1, 1))
        self._stage.write_attribute(
            self._cable_points_query,
            "omni:xform",
            ordinal=self._current_ordinal,
            tensors=xform_tensor_from_numpy(identity_xforms),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
        ).wait()

        self._allocate_cable_device_buffers(flat_shape_ids, offsets, counts)
        # The descriptors alias these slices, so both must outlive every write that uses them.
        self._cable_point_slices = [
            self._cable_points[offset + curve : offset + curve + segment_count + 1]
            for curve, (offset, segment_count) in enumerate(zip(offsets, counts, strict=True))
        ]
        self._cable_point_tensors = [points_tensor_from_warp(points) for points in self._cable_point_slices]

    def _setup_particle_bindings_ovstage(self) -> None:
        """Setup OVRTX bindings for Newton particle clouds (ovstage path)."""
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            logger.debug("NewtonManager not available, skipping particle point bindings")
            return

        particle_visual_prims = NewtonManager._particle_visual_prims
        if not particle_visual_prims:
            logger.debug("No particle visual prims registered, skipping particle point bindings")
            return

        self._particle_visual_offsets = []
        self._particle_visual_counts = []
        points_prim_paths: list[str] = []

        for prim_path, record in particle_visual_prims.items():
            points_prim_paths.append(prim_path)
            self._particle_visual_offsets.append(record.offset)
            self._particle_visual_counts.append(record.count)

        prim_count = len(points_prim_paths)

        self._particle_paths_list = self._stage_paths.create_path_list_from_strings(points_prim_paths)
        self._particle_points_query = self._stage.query_from_path_list(self._particle_paths_list)

        # Divergence from the legacy path: ovstage's PrimMode has only UPSERT and INSERT, with no
        # MUST_EXIST equivalent, so the writes below cannot assert that every registered particle
        # path resolves to a real prim the way _setup_particle_bindings_legacy does. A stale or
        # mistyped path silently upserts a new row instead of raising, surfacing as invisible
        # particles rather than an error. Same caveat applies to _setup_deformable_bindings_ovstage.
        #
        # particle_q is already in world space, so resetting the xform stack and pinning an identity
        # omni:xform prevents the env-root and asset-root ancestor transforms from being applied on top.
        self._stage.write_attribute(
            self._particle_points_query,
            "omni:resetXformStack",
            ordinal=self._current_ordinal,
            tensors=np.full(prim_count, True, dtype=np.bool_),
            is_array=False,
        ).wait()

        identity_xforms = np.tile(np.eye(4, dtype=np.float64), (prim_count, 1, 1))
        self._stage.write_attribute(
            self._particle_points_query,
            "omni:xform",
            ordinal=self._current_ordinal,
            tensors=xform_tensor_from_numpy(identity_xforms),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
        ).wait()

        if self._particle_points_query is None:
            raise RuntimeError("Failed to create OVRTX particle point bindings")

    def _update_transforms_ovstage(self) -> None:
        if self._object_xform_query is None or self._object_newton_indices is None or self._object_scales is None:
            return

        # If self._object_newton_indices is not None, then Newton's the current physics backend

        from isaaclab_newton.physics import NewtonManager

        newton_state = NewtonManager.get_state()
        if newton_state is None:
            raise RuntimeError("Newton state should not be None")

        body_q = getattr(newton_state, "body_q", None)
        if body_q is None:
            return

        num_objects = len(self._object_newton_indices)
        object_transforms = wp.empty(num_objects, dtype=wp.mat44d, device=self._device)
        wp.launch(
            kernel=sync_newton_transforms_kernel,
            dim=num_objects,
            inputs=[object_transforms, self._object_newton_indices, body_q, self._object_scales],
            device=self._device,
        )
        # The tensor is handed over zero-copy, so ovstage reads ``object_transforms`` in place and
        # must not do so until the kernel above has landed. Passing the producing Warp stream as
        # ``cuda_stream`` gives producer ordering: ovstage drains the work already queued on that
        # stream before it touches the tensor. That replaces the device-wide
        # ``wp.synchronize_device()`` with stream-scoped ordering and removes the host copy; it is
        # not a nonblocking handoff, and the ``.wait()`` below can still block the calling thread.
        # A GPU-side wait would need the event-based API instead.
        self._stage.write_attribute(
            self._object_xform_query,
            "omni:xform",
            ordinal=self._current_ordinal,
            tensors=xform_tensor_from_warp(object_transforms),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            cuda_stream=self._warp_device.stream.cuda_stream,
        ).wait()

    def _update_geometries_ovstage(self) -> None:
        if self._deformable_points_query is not None or self._particle_points_query is not None:
            # If either geometry query is not None, then Newton's the current physics backend
            from isaaclab_newton.physics import NewtonManager

            newton_state = NewtonManager.get_state()
            if newton_state is None:
                raise RuntimeError("Newton state should not be None")

            # particle_q is the world-space particle positions for all deformable bodies and particle
            # clouds. A non-None geometry query means entries were registered, so Newton must
            # expose particle state; a missing particle_q here is an inconsistent state.
            particle_q = getattr(newton_state, "particle_q", None)
            if particle_q is None:
                raise RuntimeError("Newton state has no particle_q but particle geometry queries exist")

            if self._deformable_points_query is not None:
                self._write_particle_q_slices_ovstage(
                    self._deformable_points_query,
                    particle_q,
                    self._deformable_particle_offsets,
                    self._deformable_particle_counts,
                )

            if self._particle_points_query is not None:
                self._write_particle_q_slices_ovstage(
                    self._particle_points_query,
                    particle_q,
                    self._particle_visual_offsets,
                    self._particle_visual_counts,
                )

        if self._cable_points_query is not None:
            self._write_cable_points_ovstage()

    def _write_particle_q_slices_ovstage(
        self,
        query: Any,
        particle_q: wp.array,
        particle_offsets: list[int],
        particle_counts: list[int],
    ) -> None:
        """Write world-space ``particle_q`` slices into the ``points`` column of one ovstage query.

        Args:
            query: ovstage query selecting the prims whose ``points`` attribute is written.
            particle_q: Flat world-space particle positions [m], shape ``[total_particles]``,
                dtype ``wp.vec3f``. Slices are passed zero-copy as CUDA DLTensors.
            particle_offsets: Start index of each prim's slice into :paramref:`particle_q`.
            particle_counts: Number of particles in each prim's slice.
        """
        particle_slices = [
            points_tensor_from_warp(particle_q[particle_offset : particle_offset + particle_count])
            for particle_offset, particle_count in zip(particle_offsets, particle_counts, strict=True)
        ]

        # The slices alias ``particle_q`` and are handed over zero-copy, so ovstage must not read
        # them until the Warp kernels that wrote ``particle_q`` have finished. Passing the producing
        # Warp stream as ``cuda_stream`` gives producer ordering: ovstage drains the work already
        # queued on that stream before it touches the slices. That replaces the device-wide
        # ``wp.synchronize_device()`` with stream-scoped ordering and removes the host copy; it is
        # not a nonblocking handoff, and the ``.wait()`` below can still block the calling thread.
        self._stage.write_attribute(
            query,
            "points",
            ordinal=self._current_ordinal,
            tensors=particle_slices,
            is_array=True,
            semantic=ovstage.AttributeSemantic.POINT,
            cuda_stream=self._warp_device.stream.cuda_stream,
        ).wait()

    def _write_cable_points_ovstage(self) -> None:
        """Recompute world-space cable curve points on device and write them through ovstage."""
        self._compute_cable_points_world()

        # The cached descriptors alias ``_cable_points`` and are handed over zero-copy, so ovstage
        # must not read them until the kernel above has landed. Passing the producing Warp stream as
        # ``cuda_stream`` gives producer ordering: ovstage drains the work already queued on that
        # stream before it touches the slices. That keeps the handover off the host; it is not a
        # nonblocking handoff, and the ``.wait()`` below can still block the calling thread.
        self._stage.write_attribute(
            self._cable_points_query,
            "points",
            ordinal=self._current_ordinal,
            tensors=self._cable_point_tensors,
            is_array=True,
            semantic=ovstage.AttributeSemantic.POINT,
            cuda_stream=self._warp_device.stream.cuda_stream,
        ).wait()

    def _update_camera_ovstage(
        self,
        render_data: OVRTXRenderData,
        positions: ProxyArray,
        orientations: ProxyArray,
        intrinsics: ProxyArray,
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
        if self._camera_xform_query is not None:
            # Stream-ordered zero-copy handoff, as for the object transforms above.
            self._stage.write_attribute(
                self._camera_xform_query,
                "omni:xform",
                ordinal=self._current_ordinal,
                tensors=xform_tensor_from_warp(camera_transforms),
                is_array=False,
                semantic=ovstage.AttributeSemantic.MATRIX,
                cuda_stream=self._warp_device.stream.cuda_stream,
            ).wait()

    def _render_ovstage(self, render_data: OVRTXRenderData) -> None:
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call initialize() first.")
        if self._renderer is None or len(self._render_product_paths) == 0:
            return
        # Commit all per-frame writes (transforms, geometries, camera, materials) then step.
        # advance_write_floor must precede step — the renderer rejects ordinal > write_floor.
        material_writer = self._visual_material_writer_ref() if self._visual_material_writer_ref is not None else None
        try:
            if material_writer is not None:
                material_writer.publish()
            self._stage.advance_write_floor(ordinal=self._current_ordinal).wait()
        finally:
            if material_writer is not None:
                drain_errors = contextlib.nullcontext() if sys.exc_info()[0] is None else contextlib.suppress(Exception)
                with drain_errors:
                    material_writer.drain()
        products = self._renderer.step(
            render_products=set(self._render_product_paths),
            delta_time=1.0 / 60.0,
            ordinal=self._current_ordinal,
        )
        self._current_ordinal += 1
        product_path = self._render_product_paths[0]
        if product_path in products and len(products[product_path].frames) > 0:
            self._process_render_frame(
                render_data,
                products[product_path].frames[0],
                render_data.warp_buffers,
            )

        # Post-render PPISP: HDR scene-linear → LDR RGBA. Source/destination
        # buffers are the same warp buffer map used by extraction.
        if render_data.ppisp_pipeline is not None:
            render_data.ppisp_pipeline.apply(
                render_data.warp_buffers[str(RenderBufferKind.RGB_HDR)],
                render_data.warp_buffers[str(RenderBufferKind.RGBA)],
            )

    def _close_ovstage(self) -> None:
        """Release the renderer's stage queries, path lists and ovstage stage. See :meth:`close`."""

        def _safe_release_query(query, name: str) -> None:
            if query is None or self._stage is None:
                return
            try:
                self._stage.release_query(query).wait()
            except Exception as e:
                if "destroyed" not in str(e).lower():
                    logger.warning("Error releasing %s query: %s", name, e)

        def _safe_destroy_path_list(path_list, name: str) -> None:
            if path_list is None or self._stage_paths is None:
                return
            try:
                self._stage_paths.destroy_path_list(path_list)
            except Exception as e:
                if "destroyed" not in str(e).lower():
                    logger.warning("Error destroying %s path list: %s", name, e)

        _safe_release_query(self._camera_xform_query, "camera transforms")
        self._camera_xform_query = None
        _safe_destroy_path_list(self._camera_paths_list, "camera paths")
        self._camera_paths_list = None
        _safe_release_query(self._object_xform_query, "object transforms")
        self._object_xform_query = None
        _safe_destroy_path_list(self._object_paths_list, "object paths")
        self._object_paths_list = None
        _safe_release_query(self._deformable_points_query, "deformable points")
        self._deformable_points_query = None
        _safe_destroy_path_list(self._deformable_paths_list, "deformable paths")
        self._deformable_paths_list = None
        _safe_release_query(self._particle_points_query, "particle points")
        self._particle_points_query = None
        _safe_destroy_path_list(self._particle_paths_list, "particle paths")
        self._particle_paths_list = None

        _safe_release_query(self._cable_points_query, "cable points")
        self._cable_points_query = None
        _safe_destroy_path_list(self._cable_paths_list, "cable paths")
        self._cable_paths_list = None

        self._object_newton_indices = None
        self._object_scales = None
        self._object_scales_by_path = {}
        self._deformable_particle_offsets = []
        self._deformable_particle_counts = []
        self._particle_visual_offsets = []
        self._particle_visual_counts = []
        self._cable_segment_counts = []
        self._cable_max_points = 0
        # Descriptors alias ``_cable_points``; drop them before the buffer so no cached
        # DLTensor can outlive the device memory it points at.
        self._cable_point_tensors = []
        self._cable_point_slices = []
        self._cable_points = None
        self._cable_shape_ids = None
        self._cable_offsets = None
        self._cable_counts = None

        # Detach before closing ExitStack: the renderer holds a live reference into the stage,
        # so detaching first avoids a use-after-free when ExitStack destroys Stage and PathDictionary.
        #
        # Both are guarded because close() can run before initialization completed — the sim was
        # never played, or scene setup raised — and it must stay a no-op when called twice.
        # detach_ovstage() raises when nothing is attached, and the ExitStack does not exist until
        # _initialize_from_spec_ovstage creates it.
        if self._renderer is not None and self._stage is not None:
            self._renderer.detach_ovstage()
        self._renderer = None

        if self._ovstage_exit_stack is not None:
            self._ovstage_exit_stack.close()
        self._ovstage_exit_stack = None
        self._stage = None
        self._stage_paths = None

        self._render_product_paths.clear()
        self._output_id_color_buffers.clear()
        self._initialized_scene = False
        self._current_ordinal = 0
