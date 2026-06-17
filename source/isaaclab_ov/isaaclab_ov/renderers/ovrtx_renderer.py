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
from itertools import compress
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

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
    from ovrtx import Device, PrimMode, Renderer, RendererConfig, Semantic
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

from .ovrtx_renderer_cfg import OVRTXRendererCfg
from .ovrtx_renderer_kernels import (
    create_camera_transforms_kernel,
    extract_all_depth_tiles_kernel,
    extract_all_rgb_float_tiles_kernel,
    extract_all_rgb_half_tiles_kernel,
    extract_all_rgba_tiles_kernel,
    generate_random_colors_from_ids_kernel,
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


def _raise_missing_ppisp_error(exc: ModuleNotFoundError) -> NoReturn:
    # Only translate missing isaaclab_ppisp imports into the optional-dependency hint;
    # unrelated missing modules should surface unchanged for easier debugging.
    if exc.name != "isaaclab_ppisp" and not (exc.name and exc.name.startswith("isaaclab_ppisp.")):
        raise exc
    raise ModuleNotFoundError(_PPISP_IMPORT_ERROR_MESSAGE, name="isaaclab_ppisp") from exc


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
        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.RGB_HDR: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.ALBEDO: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SIMPLE_SHADING_FULL_MDL: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.SEMANTIC_SEGMENTATION: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
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
        self._initialized_scene = False
        self._exported_usd_string: str | None = None
        self._camera_rel_path: str | None = None
        self._output_semantic_color_buffer: wp.array | None = None
        self._clone_plan: ClonePlan | None = None

        logger.info("Creating OVRTX renderer...")
        OVRTX_CONFIG = RendererConfig(
            log_file_path=self.cfg.log_file_path,
            log_level=self.cfg.log_level,
            read_gpu_transforms=True,
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

        if self._exported_usd_string is not None:
            logger.info("Injecting camera definitions...")

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
            try:
                self._renderer.open_usd_from_string(combined_usd_string)
                logger.info("OVRTX loaded USD from string successfully")
            except Exception as e:
                logger.exception("Error loading USD: %s", e)
                raise

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
            try:
                self._renderer.write_attribute(
                    prim_paths=camera_paths,
                    attribute_name="omni:resetXformStack",
                    tensor=np.full(num_envs, True, dtype=np.bool_),
                )
            except Exception as e:
                logger.warning("Failed to write omni:resetXformStack: %s", e)

            if self._camera_binding is not None:
                logger.info("Camera binding created successfully")
            else:
                logger.warning("Camera binding is None")

            self._setup_object_bindings()

    def _clone_sources_in_ovrtx(self):
        """Clone sources in OVRTX using the scene :class:`~isaaclab.cloner.ClonePlan`."""
        clone_plan = self._clone_plan
        if clone_plan is None:
            raise RuntimeError("Clone plan is required when using OVRTX cloning")

        logger.info("Cloning sources in OVRTX...")

        num_envs = clone_plan.clone_mask.shape[1]
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

    def _update_scene_partitions_after_clone(self, num_envs: int):
        """Update scene partition attributes on cloned environments and cameras in OvRTX."""
        logger.info("Writing scene partitions for %d environments...", num_envs)
        partition_tokens = [f"env_{i}" for i in range(num_envs)]
        env_prim_paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
        camera_prim_paths = [f"/World/envs/env_{i}/{self._camera_rel_path}" for i in range(num_envs)]

        try:
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
        except Exception as e:
            logger.warning("Failed to write scene partitions: %s", e, exc_info=True)

    def _setup_object_bindings(self):
        """Setup OVRTX bindings for scene objects to sync with Newton physics."""
        try:
            from isaaclab_newton.physics import NewtonManager

            newton_model = NewtonManager.get_model()
            if newton_model is None:
                logger.info("Newton model not available, skipping object bindings")
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

            try:
                self._renderer.write_attribute(
                    prim_paths=object_paths,
                    attribute_name="omni:resetXformStack",
                    tensor=np.full(len(object_paths), True, dtype=np.bool_),
                )
            except Exception as e:
                logger.warning("Failed to write omni:resetXformStack on objects: %s", e)

            if self._object_binding is not None:
                logger.info("Object binding created successfully")
                self._object_newton_indices = wp.array(newton_indices, dtype=wp.int32, device=self._device)
            else:
                logger.warning("Object binding is None")
        except ImportError:
            logger.info("Newton not available, skipping object bindings")
        except Exception as e:
            logger.warning("Error setting up object bindings: %s", e)

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

    def update_transforms(self) -> None:
        """Sync physics objects to OVRTX."""
        if self._object_binding is None or self._object_newton_indices is None:
            return

        try:
            from isaaclab_newton.physics import NewtonManager

            newton_state = NewtonManager.get_state()
            if newton_state is None:
                return
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
        except Exception as e:
            logger.warning("Failed to update object transforms: %s", e)

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
        """No-op: outputs already live in the caller's torch storage.

        :meth:`set_outputs` wraps each ``camera_data.output`` tensor as a
        zero-copy warp array stored in ``render_data.warp_buffers``, and
        :meth:`render` writes the rendered tiles directly into those warp
        arrays. There is therefore nothing to copy here.

        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.read_output`.
        """

    def _generate_random_colors_from_ids(self, input_ids: wp.array) -> wp.array:
        """Generate pseudo-random colors from semantic IDs."""
        if self._output_semantic_color_buffer is None or self._output_semantic_color_buffer.shape != input_ids.shape:
            self._output_semantic_color_buffer = wp.zeros(shape=input_ids.shape, dtype=wp.uint32, device=self._device)

        output_colors = self._output_semantic_color_buffer

        wp.launch(
            kernel=generate_random_colors_from_ids_kernel,
            dim=input_ids.shape,
            inputs=[input_ids, output_colors],
            device=self._device,
        )

        return output_colors

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

        wp.launch(
            kernel=extract_all_rgba_tiles_kernel,
            dim=(render_data.num_envs, render_data.height, render_data.width),
            inputs=[
                tiled_data,
                output_buffer,
                render_data.num_cols,
                render_data.width,
                render_data.height,
                num_channels,
            ],
            device=self._device,
        )

    def _extract_depth_tiles(
        self, render_data: OVRTXRenderData, tiled_depth_data: wp.array, output_buffers: dict
    ) -> None:
        """Extract per-env depth tiles into output_buffers (single kernel launch)."""
        for depth_type in ["depth", "distance_to_image_plane", "distance_to_camera"]:
            if depth_type in output_buffers:
                wp.launch(
                    kernel=extract_all_depth_tiles_kernel,
                    dim=(render_data.num_envs, render_data.height, render_data.width),
                    inputs=[
                        tiled_depth_data,
                        output_buffers[depth_type],
                        render_data.num_cols,
                        render_data.width,
                        render_data.height,
                    ],
                    device=self._device,
                )

    def _extract_hdr_color_tiles(
        self, render_data: OVRTXRenderData, tiled_data: wp.array, output_buffers: dict
    ) -> None:
        """Extract per-env HdrColor tiles into output_buffers."""
        if "rgb_hdr" not in output_buffers:
            return
        if tiled_data.dtype == wp.float16:
            kernel = extract_all_rgb_half_tiles_kernel
        elif tiled_data.dtype == wp.float32:
            kernel = extract_all_rgb_float_tiles_kernel
        else:
            raise TypeError(f"Unsupported OVRTX HdrColor dtype: {tiled_data.dtype}.")
        wp.launch(
            kernel=kernel,
            dim=(render_data.num_envs, render_data.height, render_data.width),
            inputs=[
                tiled_data,
                output_buffers["rgb_hdr"],
                render_data.num_cols,
                render_data.width,
                render_data.height,
            ],
            device=self._device,
        )

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

        for depth_var in ["DistanceToImagePlaneSD", "DepthSD"]:
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

        if "SemanticSegmentation" in frame.render_vars and "semantic_segmentation" in output_buffers:
            with frame.render_vars["SemanticSegmentation"].map(device=Device.CUDA) as mapping:
                tiled_semantic_data = wp.from_dlpack(mapping.tensor)

                if tiled_semantic_data.dtype == wp.uint32:
                    semantic_colors = self._generate_random_colors_from_ids(tiled_semantic_data)

                    semantic_torch = wp.to_torch(semantic_colors)
                    semantic_uint8 = semantic_torch.view(torch.uint8)

                    if semantic_torch.dim() == 2:
                        h, w = semantic_torch.shape
                        semantic_uint8 = semantic_uint8.reshape(h, w, 4)

                    tiled_semantic_data = wp.from_torch(semantic_uint8, dtype=wp.uint8)

                self._extract_rgba_tiles(
                    render_data,
                    tiled_semantic_data,
                    output_buffers,
                    "semantic_segmentation",
                    suffix="semantic",
                )

    def render(self, render_data: OVRTXRenderData) -> None:
        """Render the scene into the provided RenderData."""
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call initialize() first.")
        if self._renderer is None or len(self._render_product_paths) == 0:
            return
        try:
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
        except Exception as e:
            logger.warning("OVRTX rendering failed: %s", e, exc_info=True)

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

        if self._renderer:
            try:
                self._renderer.reset_stage()
            except Exception as e:
                logger.warning("Error resetting stage: %s", e)

            self._renderer = None

        self._render_product_paths.clear()
        self._output_semantic_color_buffer = None
        self._initialized_scene = False
