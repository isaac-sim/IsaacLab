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

import logging
import math
import os
import re
from itertools import compress
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

logger = logging.getLogger(__name__)

import numpy as np
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
    )
except ModuleNotFoundError as exc:
    if exc.name != "ovrtx":
        raise
    raise ModuleNotFoundError(
        "The OVRTX renderer requires the optional 'ovrtx' runtime wheel, which is not installed. "
        "Install it with: ./isaaclab.sh -i 'ov[ovrtx]' "
        "(or, manually: pip install --extra-index-url https://pypi.nvidia.com -e 'source/isaaclab_ov[ovrtx]')."
    ) from exc

from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp

from .ovrtx_annotator_utils import build_semantic_id_to_labels, decode_semantic_id_map
from .ovrtx_renderer_cfg import OVRTXRendererCfg
from .ovrtx_renderer_kernels import (
    compute_deformable_mesh_extent_kernel,
    create_camera_transforms_kernel,
    extract_all_tiles_kernel,
    generate_random_colors_from_ids_kernel,
    sync_newton_deformable_points_batched_kernel,
    sync_newton_transforms_kernel,
)
from .ovrtx_usd import (
    build_render_product_as_string,
    create_scene_partition_attributes,
    export_stage_to_string,
)

if TYPE_CHECKING:
    from isaaclab_ppisp import PpispPipeline

    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

from isaaclab.renderers.camera_render_spec import CameraRenderSpec

# The resolved integer value is assigned to the ``omni:rtx:minimal:mode`` attribute of the render product.
_RTX_MINIMAL_MODES = {
    RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE.value: 1,
    RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL.value: 2,
    RenderBufferKind.SIMPLE_SHADING_FULL_MDL.value: 3,
}

_PPISP_IMPORT_ERROR_MESSAGE = (
    "isaaclab_ppisp is required when CameraCfg.isp_cfg is set. "
    "Install Isaac Lab with the 'all' extra (`pip install isaaclab[all]`) or install the "
    "isaaclab-ppisp extension from the Isaac Lab source checkout."
)
_READ_GPU_TRANSFORMS_ENV = "ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS"


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


def _create_homogeneous_clone_plan(num_envs: int) -> ClonePlan:
    """Create a homogeneous fallback plan that replicates ``env_0`` to every environment."""
    return ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, num_envs), dtype=torch.bool, device="cpu"),
    )


def _resolve_clone_plan(num_envs: int) -> ClonePlan:
    """Resolve clone plan for local use.

    If no clone plan is published by the scene or it has no active rows, returns a homogeneous clone plan.
    If the clone plan has some inactive rows, returns a copy of the published clone plan with only the active rows.
    If the clone plan has all active rows, returns the published clone plan (shallow copy).
    """
    published_clone_plan = SimulationContext.instance().get_clone_plan()

    if published_clone_plan is None:
        logger.warning("No clone plan is published by the scene; returning homogeneous clone plan")
        return _create_homogeneous_clone_plan(num_envs)

    active_rows = published_clone_plan.clone_mask.any(dim=1)

    # If no rows are active, return a homogeneous clone plan.
    if not active_rows.any():
        logger.warning("Clone plan has no active rows, returning homogeneous clone plan")
        return _create_homogeneous_clone_plan(num_envs)

    # If some rows are inactive, return a copy of the published clone plan with only the active rows.
    if not active_rows.all():
        logger.warning("Clone plan has some inactive rows; returning a copy with only active rows")
        active = active_rows.tolist()
        return ClonePlan(
            sources=tuple(compress(published_clone_plan.sources, active)),
            destinations=tuple(compress(published_clone_plan.destinations, active)),
            clone_mask=published_clone_plan.clone_mask[active_rows],
        )

    # If all rows are active, return the published clone plan (shallow copy).
    return published_clone_plan


def _gf_matrix_to_wp_mat44f(matrix: Any) -> wp.mat44f:
    """Convert a USD ``Gf.Matrix4d``-like object to a Warp ``mat44f``."""
    # ``Gf.Matrix4d`` indexes by row, while Warp matrix constructors take values
    # in column-major order for ``wp.transform_point`` semantics.
    return wp.mat44f(*[float(matrix[row][col]) for col in range(4) for row in range(4)])


def _clone_plan_env_ids(clone_plan: ClonePlan) -> torch.Tensor:
    """Return the environment ids addressed by a clone plan."""
    if clone_plan.env_ids is not None:
        return clone_plan.env_ids.to(device=clone_plan.clone_mask.device)
    return torch.arange(clone_plan.clone_mask.shape[1], dtype=torch.long, device=clone_plan.clone_mask.device)


def _resolve_deformable_instance_path(path: str, instance_index: int) -> str:
    """Resolve a deformable registry path template for one cloned instance."""
    resolved_path = re.sub(r"(?<=[Ee]nv_)\.\*", str(instance_index), path)
    return re.sub(r"\.\*", str(instance_index), resolved_path)


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
        # Currently only "semantic_segmentation" is populated (with an "idToLabels" mapping).
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

    @classmethod
    def provides_temporal_camera_data(cls, data_type: str) -> bool:
        # OV-RTX, like Isaac RTX, temporally accumulates only the rgb/rgba beauty buffer (DLSS).
        return data_type in ("rgb", "rgba")

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output layout this OVRTX backend writes.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.supported_output_types`."""
        instance_seg_spec = (
            RenderBufferSpec(4, wp.uint8) if self.cfg.colorize_instance_segmentation else RenderBufferSpec(1, wp.uint32)
        )
        instance_id_seg_spec = (
            RenderBufferSpec(4, wp.uint8)
            if self.cfg.colorize_instance_id_segmentation
            else RenderBufferSpec(1, wp.uint32)
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
            RenderBufferKind.INSTANCE_SEGMENTATION_FAST: instance_seg_spec,
            RenderBufferKind.INSTANCE_ID_SEGMENTATION_FAST: instance_id_seg_spec,
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.MOTION_VECTORS: RenderBufferSpec(2, wp.float32),
        }

    @property
    def _device_id(self) -> int:
        """CUDA device index extracted from ``self._device`` for OVRTX ``binding.map()`` calls."""
        parts = self._device.split(":")
        return int(parts[1]) if len(parts) > 1 else 0

    def __init__(self, cfg: OVRTXRendererCfg):
        self.cfg = cfg
        self._device = "cuda:0"  # default; overridden by create_render_data(spec)
        self._render_product_paths = []
        self._camera_binding = None
        self._object_binding = None
        self._object_newton_indices: wp.array | None = None
        self._deformable_point_binding = None
        self._deformable_extent_binding = None
        self._deformable_point_buffers: list[wp.array] = []
        self._deformable_extent_buffer: np.ndarray | None = None
        self._deformable_visual_mesh_paths: list[str] = []
        self._deformable_particle_offsets: wp.array | None = None
        self._deformable_inverse_world_matrices: wp.array | None = None
        self._deformable_particles_per_mesh: wp.array | None = None
        self._deformable_max_particles_per_mesh: int = 0
        self._deformable_extent_mins: list[wp.array] = []
        self._deformable_extent_maxs: list[wp.array] = []
        self._deformable_bindings_ready = False
        self._deformable_bindings_warned = False
        self._initialized_scene = False
        self._exported_usd_string: str | None = None
        self._camera_rel_path: str | None = None
        self._output_id_color_buffers: dict[str, wp.array] = {}
        self._clone_plan: ClonePlan | None = None

        logger.info("Creating OVRTX renderer...")
        OVRTX_CONFIG = RendererConfig(
            log_file_path=self.cfg.log_file_path,
            log_level=self.cfg.log_level,
            read_gpu_transforms=_read_gpu_transforms_enabled(),
            keep_system_alive=True,
        )
        self._renderer = Renderer(OVRTX_CONFIG)
        if not self._renderer:
            raise RuntimeError(
                "Failed to create OVRTX Renderer; the underlying ovrtx.Renderer constructor returned a falsy"
                " value. Check that ovrtx is installed correctly and its native dependencies are available."
            )
        logger.info("OVRTX renderer created successfully")

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

        # If temp_usd_dir is set, write the pre-ovrtx stage to a temporary file.
        if self.cfg.temp_usd_dir is not None:
            _write_file(Path(self.cfg.temp_usd_dir), "pre_ovrtx_renderer_stage.usda", stage.ExportToString())

        logger.info("Preparing stage (%d envs)...", num_envs)
        create_scene_partition_attributes(stage, num_envs)

        # Resolve the clone plan for local use.
        self._clone_plan = _resolve_clone_plan(num_envs)
        if self._clone_plan is None:
            raise RuntimeError("Clone plan is required when preparing OVRTX stage")

        self._exported_usd_string = export_stage_to_string(
            stage,
            num_envs,
            source_paths=self._clone_plan.sources,
        )

    def _initialize_from_spec(self, spec: CameraRenderSpec):
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

        if num_envs > 1:
            self._clone_sources_in_ovrtx()
            self._update_scene_partitions_after_clone(num_envs)

        self._initialized_scene = True

        camera_paths = [f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(num_envs)]
        self._camera_binding = self._renderer.bind_attribute(
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

        if self._camera_binding is not None:
            logger.info("Camera binding created successfully")
        else:
            raise RuntimeError("Camera binding is None — cannot render without a valid camera binding")

        self._setup_object_bindings()
        self._setup_deformable_mesh_bindings()

    def _clone_sources_in_ovrtx(self):
        """Clone sources in OVRTX using the scene :class:`~isaaclab.cloner.ClonePlan`."""
        clone_plan = self._clone_plan
        if clone_plan is None:
            raise RuntimeError("Clone plan is required when using OVRTX cloning")

        num_envs = clone_plan.clone_mask.shape[1]
        env_prim_paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
        xform_attr_name = "omni:xform"

        # Snapshot per-env root transforms before clone_usd overwrites them with the source env root.
        env_root_xforms = np.empty((num_envs, 4, 4), dtype=np.float64)
        self._renderer.read_attribute(
            xform_attr_name,
            env_prim_paths,
            prim_mode=PrimMode.MUST_EXIST,
            dest=env_root_xforms,
        )
        logger.info("Captured per-env root transforms before cloning")

        logger.info("Cloning sources in OVRTX...")
        env_ids = torch.arange(num_envs, dtype=torch.int32, device=clone_plan.clone_mask.device)

        num_cloned_sources = 0

        for row_idx, (source, destination) in enumerate(zip(clone_plan.sources, clone_plan.destinations, strict=True)):
            target_env_ids = env_ids[clone_plan.clone_mask[row_idx]].tolist()

            target_paths = []
            for env_id in target_env_ids:
                resolved_destination = destination.format(env_id)
                if resolved_destination != source:
                    target_paths.append(resolved_destination)

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

        # OVRTX clones the source env root transform verbatim. Restore per-env root transforms so any objects that
        # are not covered by body labels still inherit the correct clone origin.
        self._renderer.write_attribute(
            prim_paths=env_prim_paths,
            attribute_name=xform_attr_name,
            tensor=env_root_xforms,
            semantic=Semantic.XFORM_MAT4x4,
            prim_mode=PrimMode.MUST_EXIST,
        )
        logger.info("Restored per-env root transforms after cloning", num_envs)

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

    def _setup_object_bindings(self):
        """Setup OVRTX bindings for scene objects to sync with Newton physics."""
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            logger.debug("isaaclab_newton physics not available, skipping object bindings")
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

        self._object_binding = self._renderer.bind_attribute(
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

        if self._object_binding is None:
            raise RuntimeError("Failed to create OVRTX object bindings")

        self._object_newton_indices = wp.array(newton_indices, dtype=wp.int32, device=self._device)

    def _setup_deformable_mesh_bindings(self):
        """Setup OVRTX mesh ``points`` bindings for Newton deformables."""
        self._deformable_bindings_ready = False
        try:
            from isaaclab_newton.physics import NewtonManager

            from pxr import UsdGeom

            from isaaclab.sim.utils.stage import get_current_stage
        except ImportError:
            logger.info("Newton/USD not available, skipping deformable mesh bindings")
            return

        deformable_registry = NewtonManager._deformable_registry
        if not deformable_registry:
            return

        renderable_entries = [
            entry for entry in deformable_registry if getattr(entry, "deformable_type", None) in ("surface", "volume")
        ]
        if not renderable_entries:
            return

        self._deformable_point_buffers = []
        self._deformable_extent_buffer = None
        self._deformable_visual_mesh_paths = []
        self._deformable_particle_offsets = None
        self._deformable_inverse_world_matrices = None
        self._deformable_particles_per_mesh = None
        self._deformable_max_particles_per_mesh = 0
        self._deformable_extent_mins = []
        self._deformable_extent_maxs = []

        stage = get_current_stage()
        if stage is None:
            prim_paths = [entry.prim_path for entry in renderable_entries]
            logger.warning(
                "Current USD stage unavailable, skipping OVRTX deformable mesh bindings for: %s",
                ", ".join(prim_paths),
            )
            return

        visual_mesh_paths: list[str] = []
        particle_offsets: list[int] = []
        particles_per_mesh: list[int] = []
        inverse_world_matrices: list[wp.mat44f] = []
        point_buffers: list[wp.array] = []
        original_extents: list[tuple[list[float], list[float]]] = []
        validation_errors: list[str] = []
        xform_cache = UsdGeom.XformCache()

        for entry in renderable_entries:
            particles_per_body = int(entry.particles_per_body)
            if particles_per_body <= 0:
                validation_errors.append(f"{entry.prim_path}: invalid particles_per_body={particles_per_body}")
                continue

            for inst_idx, offset in enumerate(entry.particle_offsets):
                resolved_vis = _resolve_deformable_instance_path(entry.vis_mesh_prim_path, inst_idx)
                vis_prim = stage.GetPrimAtPath(resolved_vis)
                if not vis_prim or not vis_prim.IsValid():
                    validation_errors.append(f"{resolved_vis}: visual mesh prim not found")
                    continue

                mesh = UsdGeom.Mesh(vis_prim)
                points_attr = mesh.GetPointsAttr()
                if points_attr is None:
                    validation_errors.append(f"{resolved_vis}: mesh points attribute missing")
                    continue
                points = points_attr.Get()  # type: ignore[reportAttributeAccessIssue]
                if points is None:
                    validation_errors.append(f"{resolved_vis}: mesh points missing")
                    continue
                if len(points) != particles_per_body:
                    if entry.deformable_type == "volume":
                        validation_errors.append(
                            f"{resolved_vis}: volume visual mesh has {len(points)} points but Newton has"
                            f" {particles_per_body} tet particles; OVRTX supports volume deformables only when"
                            " the visual mesh points match Newton tet particles (separate visual embedding is"
                            " not implemented)"
                        )
                    else:
                        validation_errors.append(
                            f"{resolved_vis}: mesh has {len(points)} points but Newton has"
                            f" {particles_per_body} particles"
                        )
                    continue

                points_np = np.asarray(points, dtype=np.float32)
                extent_attr = mesh.GetExtentAttr()
                extent = extent_attr.Get() if extent_attr is not None else None  # type: ignore[reportAttributeAccessIssue]
                if extent is not None and len(extent) == 2:
                    extent_min = [float(value) for value in extent[0]]
                    extent_max = [float(value) for value in extent[1]]
                else:
                    extent_min = points_np.min(axis=0).tolist()
                    extent_max = points_np.max(axis=0).tolist()

                visual_mesh_paths.append(resolved_vis)
                particle_offsets.append(int(offset))
                particles_per_mesh.append(particles_per_body)
                original_extents.append((extent_min, extent_max))
                local_to_world = xform_cache.GetLocalToWorldTransform(vis_prim)
                inverse_world_matrices.append(
                    _gf_matrix_to_wp_mat44f(local_to_world.GetInverse())  # type: ignore[reportAttributeAccessIssue]
                )
                point_buffers.append(
                    wp.empty(particles_per_body, dtype=wp.vec3f, device=self._device)  # type: ignore[arg-type]
                )

        if validation_errors:
            raise RuntimeError(
                "Failed to setup OVRTX deformable mesh bindings:\n  - " + "\n  - ".join(validation_errors)
            )

        if not visual_mesh_paths:
            return

        renderer = self._renderer
        if renderer is None:
            return

        try:
            self._deformable_point_binding = renderer.bind_array_attribute(
                prim_paths=visual_mesh_paths,
                attribute_name="points",
                dtype=np.float32,
                shape=(3,),
                prim_mode=PrimMode.EXISTING_ONLY,
                flags=BindingFlag.OPTIMIZE,
            )
            self._deformable_extent_binding = renderer.bind_attribute(
                prim_paths=visual_mesh_paths,
                attribute_name="extent",
                dtype=np.float64,
                shape=(2, 3),
                prim_mode=PrimMode.EXISTING_ONLY,
                flags=BindingFlag.OPTIMIZE,
            )
        except Exception as e:
            self._deformable_point_binding = None
            self._deformable_extent_binding = None
            raise RuntimeError("OVRTX deformable mesh bindings are unavailable.") from e

        self._deformable_point_buffers = point_buffers
        self._deformable_max_particles_per_mesh = max(particles_per_mesh)
        self._deformable_extent_buffer = np.array(original_extents, dtype=np.float64)
        self._deformable_visual_mesh_paths = visual_mesh_paths
        self._deformable_particle_offsets = wp.array(particle_offsets, dtype=wp.int32, device=self._device)
        self._deformable_particles_per_mesh = wp.array(particles_per_mesh, dtype=wp.int32, device=self._device)
        self._deformable_inverse_world_matrices = wp.array(inverse_world_matrices, dtype=wp.mat44f, device=self._device)
        self._deformable_extent_mins = [
            wp.zeros(1, dtype=wp.vec3f, device=self._device)
            for _ in visual_mesh_paths  # type: ignore[arg-type]
        ]
        self._deformable_extent_maxs = [
            wp.zeros(1, dtype=wp.vec3f, device=self._device)
            for _ in visual_mesh_paths  # type: ignore[arg-type]
        ]
        self._deformable_bindings_ready = True
        NewtonManager._mark_particles_dirty()

        logger.info("Created OVRTX deformable mesh point binding for %d visual mesh(es).", len(visual_mesh_paths))

    def _refresh_deformable_inverse_world_matrices(self) -> None:
        """Refresh visual mesh inverse world matrices used for deformable point sync."""
        if not self._deformable_visual_mesh_paths:
            return

        try:
            from pxr import UsdGeom

            from isaaclab.sim.utils.stage import get_current_stage
        except ImportError:
            logger.info("USD not available, skipping deformable mesh world-matrix refresh")
            return

        stage = get_current_stage()
        if stage is None:
            logger.warning("Current USD stage unavailable, skipping deformable mesh world-matrix refresh")
            return

        xform_cache = UsdGeom.XformCache()
        inverse_world_matrices: list[wp.mat44f] = []
        for visual_mesh_path in self._deformable_visual_mesh_paths:
            prim = stage.GetPrimAtPath(visual_mesh_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"OVRTX deformable visual mesh disappeared from USD stage: {visual_mesh_path}")
            local_to_world = xform_cache.GetLocalToWorldTransform(prim)
            inverse_world_matrices.append(_gf_matrix_to_wp_mat44f(local_to_world.GetInverse()))

        self._deformable_inverse_world_matrices = wp.array(inverse_world_matrices, dtype=wp.mat44f, device=self._device)

    def create_render_data(self, spec: CameraRenderSpec) -> OVRTXRenderData:
        """Create OVRTX-specific RenderData with GPU buffers.

        Performs OVRTX initialization (stage export, USD load, bindings) on first call,
        matching the interface of Isaac RTX and Newton Warp which need no separate initialize().
        """
        self._device = spec.device
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

    def _update_object_transforms(self) -> None:
        """Sync Newton rigid-body transforms to OVRTX."""
        if self._object_binding is None or self._object_newton_indices is None:
            return

        # If self._object_newton_indices is not None, then Newton's the current physics backend

        from isaaclab_newton.physics import NewtonManager

        newton_state = NewtonManager.get_state()
        if newton_state is None:
            raise RuntimeError("Newton state should not be None")

        body_q = getattr(newton_state, "body_q", None)
        if body_q is None:
            return

        with self._object_binding.map(device=Device.CUDA, device_id=self._device_id) as attr_mapping:
            ovrtx_transforms = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)
            wp.launch(
                kernel=sync_newton_transforms_kernel,
                dim=len(self._object_newton_indices),
                inputs=[ovrtx_transforms, self._object_newton_indices, body_q],
                device=self._device,
            )

    def _allocate_deformable_stacked_points(self) -> wp.array:
        """Allocate a 2D buffer for batched deformable point sync."""
        return wp.zeros(
            (len(self._deformable_point_buffers), self._deformable_max_particles_per_mesh),
            dtype=wp.vec3f,
            device=self._device,
        )

    def _update_deformable_points(self) -> None:
        """Sync Newton deformable particles to OVRTX mesh point arrays."""
        try:
            from isaaclab_newton.physics import NewtonManager
        except ImportError:
            logger.debug("isaaclab_newton physics not available, skipping deformable points sync")
            return

        if not self._deformable_bindings_ready:
            if not self._deformable_bindings_warned:
                registry = NewtonManager._deformable_registry or []
                if any(entry.deformable_type in ("surface", "volume") for entry in registry):
                    logger.warning("OVRTX deformable mesh bindings are not ready; deformables may render static.")
                    self._deformable_bindings_warned = True
            return

        if (
            self._deformable_point_binding is None
            or self._deformable_particle_offsets is None
            or self._deformable_particles_per_mesh is None
            or self._deformable_inverse_world_matrices is None
            or not self._deformable_point_buffers
            or self._deformable_extent_buffer is None
        ):
            return

        if not NewtonManager.particles_dirty():
            return

        newton_state = NewtonManager.get_state()
        if newton_state is None:
            return

        particle_q = newton_state.particle_q
        if particle_q is None:
            return

        if NewtonManager.transforms_dirty():
            self._refresh_deformable_inverse_world_matrices()

        if self._deformable_inverse_world_matrices is None:
            return

        stacked_points = self._allocate_deformable_stacked_points()
        wp.launch(
            kernel=sync_newton_deformable_points_batched_kernel,
            dim=(len(self._deformable_point_buffers), self._deformable_max_particles_per_mesh),
            inputs=[
                stacked_points,
                particle_q,
                self._deformable_particle_offsets,
                self._deformable_inverse_world_matrices,
                self._deformable_particles_per_mesh,
            ],
            device=self._device,
        )
        for mesh_index, point_buffer in enumerate(self._deformable_point_buffers):
            particle_count = int(self._deformable_particles_per_mesh.numpy()[mesh_index])
            wp.copy(point_buffer, stacked_points[mesh_index, :particle_count])
            wp.launch(
                kernel=compute_deformable_mesh_extent_kernel,
                dim=1,
                inputs=[
                    point_buffer,
                    self._deformable_extent_mins[mesh_index],
                    self._deformable_extent_maxs[mesh_index],
                ],
                device=self._device,
            )
            extent_min = self._deformable_extent_mins[mesh_index].numpy()[0]
            extent_max = self._deformable_extent_maxs[mesh_index].numpy()[0]
            self._deformable_extent_buffer[mesh_index] = np.stack((extent_min, extent_max))

        if self._deformable_extent_binding is not None:
            wp.synchronize_device(self._device)
            self._deformable_extent_binding.write(cast(Any, self._deformable_extent_buffer))

        self._deformable_point_binding.write(  # type: ignore[arg-type]
            cast(Any, self._deformable_point_buffers),
            data_access=DataAccess.ASYNC,
        )

        NewtonManager.clear_particles_dirty()

    def update_transforms(self) -> None:
        """Sync physics object transforms and deformable points to OVRTX."""
        self._update_object_transforms()
        self._update_deformable_points()

    def update_camera(
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
        if self._camera_binding is not None:
            with self._camera_binding.map(device=Device.CUDA, device_id=self._device_id) as attr_mapping:
                wp_transforms_view = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)
                wp.copy(wp_transforms_view, camera_transforms)

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

    def _process_id_segmentation_render_var(
        self,
        render_data: OVRTXRenderData,
        frame,
        output_buffers: dict,
        render_var_name: str,
        buffer_key: str,
        colorize: bool,
    ) -> None:
        """Extract a uint32 ID-segmentation render var into ``output_buffers[buffer_key]``.

        Shared by ``semantic_segmentation`` (``SemanticSegmentation``), ``instance_segmentation_fast``
        (``NonStableInstanceSegmentation``), and ``instance_id_segmentation_fast`` (``InstanceSegmentationSD``),
        which only differ in the source render var, the destination buffer, and whether to colorize.

        Args:
            render_data: OVRTX render data for the current frame.
            frame: OVRTX frame holding the mapped render vars.
            output_buffers: Destination warp buffers, keyed by data type.
            render_var_name: Name of the OVRTX render var to read.
            buffer_key: Data type key into ``output_buffers``.
            colorize: If True, IDs are mapped to RGBA colors; otherwise raw uint32 IDs are copied.
        """
        if render_var_name not in frame.render_vars or buffer_key not in output_buffers:
            return

        with frame.render_vars[render_var_name].map(device=Device.CUDA) as mapping:
            tiled_data = wp.from_dlpack(mapping.tensor)
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
                # Non-colorized: ensure (TH, TW, 1) shape for the uint32 extraction kernel.
                data_torch = wp.to_torch(tiled_data)
                if data_torch.dim() == 2:
                    data_torch = data_torch.unsqueeze(-1)
                tiled_data = wp.from_torch(data_torch, dtype=wp.uint32)
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
        if "SemanticIdMap" not in frame.render_vars:
            return

        with frame.render_vars["SemanticIdMap"].map(device=Device.CPU) as mapping:
            labels_by_id = decode_semantic_id_map(np.from_dlpack(mapping.tensor))

        render_data.renderer_info["semantic_segmentation"] = {
            "idToLabels": build_semantic_id_to_labels(
                labels_by_id, colorize=self.cfg.colorize_semantic_segmentation, device=self._device
            )
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
        self, render_data: OVRTXRenderData, tiled_depth_data: wp.array, output_buffers: dict
    ) -> None:
        """Extract per-env depth tiles into output_buffers (single kernel launch)."""
        for depth_type in ["depth", "distance_to_image_plane", "distance_to_camera"]:
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

        # FIXME: OVRTX render var mapping can select a different CUDA device
        # than the camera/output buffers on MGPU systems. Keep this PPISP-only
        # bridge until render var mapping can be constrained like transform
        # bindings, which use ``device_id=self._device_id``.
        return wp.clone(tiled_data, device=output_device)

    def _process_render_frame(self, render_data: OVRTXRenderData, frame, output_buffers: dict) -> None:
        """Extract RGB, depth, albedo, and semantic from a single render frame into output_buffers."""
        # Reset per-output metadata so it is a snapshot of this frame only. Unlike pixel AOVs (always
        # present), metadata like the semantic ``idToLabels`` is only repopulated below when its render var
        # is available, so without this a missing SemanticIdMap on a later frame would leave a stale mapping.
        render_data.renderer_info.clear()

        if "LdrColor" in frame.render_vars:
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
                with frame.render_vars["LdrColor"].map(device=Device.CUDA) as mapping:
                    tiled_data = wp.from_dlpack(mapping.tensor)
                    self._extract_rgba_tiles(render_data, tiled_data, output_buffers, buffer_key)

        for depth_var in ["DistanceToCameraSD", "DistanceToImagePlaneSD", "DepthSD"]:
            if depth_var not in frame.render_vars:
                continue
            with frame.render_vars[depth_var].map(device=Device.CUDA) as mapping:
                tiled_depth_data = wp.from_dlpack(mapping.tensor)
                if tiled_depth_data.dtype == wp.uint32:
                    tiled_depth_data = wp.from_torch(
                        wp.to_torch(tiled_depth_data).view(torch.float32), dtype=wp.float32
                    )
                self._extract_depth_tiles(render_data, tiled_depth_data, output_buffers)
            break

        if "DiffuseAlbedoSD" in frame.render_vars and "albedo" in output_buffers:
            with frame.render_vars["DiffuseAlbedoSD"].map(device=Device.CUDA) as mapping:
                tiled_albedo_data = wp.from_dlpack(mapping.tensor)
                self._extract_rgba_tiles(render_data, tiled_albedo_data, output_buffers, "albedo", suffix="albedo")

        if "HdrColor" in frame.render_vars and "rgb_hdr" in output_buffers:
            with frame.render_vars["HdrColor"].map(device=Device.CUDA) as mapping:
                tiled_hdr_data = wp.from_dlpack(mapping.tensor)
                tiled_hdr_data = self._prepare_ppisp_hdr_source(render_data, tiled_hdr_data, output_buffers)
                self._extract_hdr_color_tiles(render_data, tiled_hdr_data, output_buffers)

        self._process_id_segmentation_render_var(
            render_data,
            frame,
            output_buffers,
            "SemanticSegmentation",
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
            "NonStableInstanceSegmentation",
            "instance_segmentation_fast",
            self.cfg.colorize_instance_segmentation,
        )

        self._process_id_segmentation_render_var(
            render_data,
            frame,
            output_buffers,
            "InstanceSegmentationSD",
            "instance_id_segmentation_fast",
            self.cfg.colorize_instance_id_segmentation,
        )

        if "NormalSD" in frame.render_vars and "normals" in output_buffers:
            with frame.render_vars["NormalSD"].map(device=Device.CUDA) as mapping:
                tiled_normals_data = wp.from_dlpack(mapping.tensor)
                self._launch_extract_all_tiles(render_data, tiled_normals_data, output_buffers["normals"])

        # For motion vectors, extract only the first two (u, v) channels from the tiled buffer.
        # Note: mirrors the Isaac RTX renderer's handling of the "TargetMotionSD" AOV
        # (check: https://github.com/isaac-sim/IsaacLab/issues/2003).
        if "TargetMotionSD" in frame.render_vars and "motion_vectors" in output_buffers:
            with frame.render_vars["TargetMotionSD"].map(device=Device.CUDA) as mapping:
                tiled_motion_vectors_data = wp.from_dlpack(mapping.tensor)
                self._launch_extract_all_tiles(render_data, tiled_motion_vectors_data, output_buffers["motion_vectors"])

    def render(self, render_data: OVRTXRenderData) -> None:
        """Render the scene into the provided RenderData."""
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call initialize() first.")
        if self._renderer is None or len(self._render_product_paths) == 0:
            return
        products = self._renderer.step(
            render_products=set(self._render_product_paths),
            delta_time=1.0 / 60.0,
        )
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

    def cleanup(self, render_data: OVRTXRenderData | None) -> None:
        """Release renderer resources. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""

        # Unbind before tearing down renderer
        def _safe_unbind(binding, name: str) -> None:
            if binding is None:
                return
            try:
                binding.unbind()
            except Exception as e:
                if "destroyed" not in str(e).lower():
                    logger.warning("Error unbinding %s: %s", name, e)

        _safe_unbind(self._camera_binding, "camera transforms")
        self._camera_binding = None
        _safe_unbind(self._object_binding, "object transforms")
        self._object_binding = None
        _safe_unbind(self._deformable_point_binding, "deformable mesh points")
        self._deformable_point_binding = None
        _safe_unbind(self._deformable_extent_binding, "deformable mesh extents")
        self._deformable_extent_binding = None
        self._deformable_point_buffers = []
        self._deformable_extent_buffer = None
        self._deformable_visual_mesh_paths = []
        self._deformable_particle_offsets = None
        self._deformable_inverse_world_matrices = None
        self._deformable_particles_per_mesh = None
        self._deformable_max_particles_per_mesh = 0
        self._deformable_extent_mins = []
        self._deformable_extent_maxs = []
        self._deformable_bindings_ready = False
        self._deformable_bindings_warned = False

        if self._renderer:
            try:
                self._renderer.reset_stage()
            except Exception as e:
                logger.warning("Error resetting stage: %s", e)

            self._renderer = None

        self._render_product_paths.clear()
        self._output_id_color_buffers.clear()
        self._initialized_scene = False
