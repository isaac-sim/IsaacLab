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

- **ovrtx_renderer_kernels.py**: Warp GPU kernels and DEVICE constant.

- **ovrtx_usd.py**: USD helpers for OVRTX: render var config, camera injection, etc.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

import numpy as np
import torch
import warp as wp

# The ovrtx C library links to its own version of the USD libraries. Having
# the pxr Python package available can cause the C library to load an
# incompatible version of libusd, potentially leading to undefined behavior.
# By setting OVRTX_SKIP_USD_CHECK, we prevent the C library from loading the pxr Python package.
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

import ovrtx
from ovrtx import Device, PrimMode, Renderer, RendererConfig, Semantic
from packaging.version import Version

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from .ovrtx_renderer_cfg import OVRTXRendererCfg
from .ovrtx_renderer_kernels import (
    DEVICE,
    create_camera_transforms_kernel,
    extract_all_depth_tiles_kernel,
    extract_all_depth_tiles_kernel_legacy,
    extract_all_rgba_tiles_kernel,
    generate_random_colors_from_ids_kernel,
    generate_random_colors_from_ids_kernel_legacy,
    sync_newton_transforms_kernel,
)
from .ovrtx_usd import (
    create_cloning_attributes,
    export_stage_for_ovrtx,
    inject_cameras_into_usd,
)

if TYPE_CHECKING:
    from isaaclab.sensors.camera.camera_data import CameraData

from isaaclab.renderers.camera_render_spec import CameraRenderSpec

# Shared integration floor for this module; reuse for ovrtx features that share one support floor.
_OVRTX_VERSION = Version(ovrtx.__version__)
_IS_OVRTX_0_3_0_OR_NEWER = Version("0.3.0") <= _OVRTX_VERSION

# The resolved integer value is assigned to the ``omni:rtx:minimal:mode`` attribute of the render product.
_RTX_MINIMAL_MODES = {
    RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE.value: 1,
    RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL.value: 2,
    RenderBufferKind.SIMPLE_SHADING_FULL_MDL.value: 3,
}


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
            RenderBufferKind.RGBA: RenderBufferSpec(4, torch.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, torch.uint8),
            RenderBufferKind.ALBEDO: RenderBufferSpec(4, torch.uint8),
            RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE: RenderBufferSpec(3, torch.uint8),
            RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL: RenderBufferSpec(3, torch.uint8),
            RenderBufferKind.SIMPLE_SHADING_FULL_MDL: RenderBufferSpec(3, torch.uint8),
            RenderBufferKind.SEMANTIC_SEGMENTATION: RenderBufferSpec(4, torch.uint8),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, torch.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, torch.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, torch.float32),
        }

    def __init__(self, cfg: OVRTXRendererCfg):
        self.cfg = cfg
        self._usd_handles = []
        self._render_product_paths = []
        self._camera_binding = None
        self._object_binding = None
        self._object_newton_indices: wp.array | None = None
        self._initialized_scene = False
        self._exported_usd_path: str | None = None
        self._camera_rel_path: str | None = None
        self._output_semantic_color_buffer: wp.array | None = None

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Export the USD stage for OVRTX before create_render_data.

        Adds cloning attributes and exports the stage to a temporary file.
        The exported path is used by create_render_data when loading into OVRTX.
        """
        if stage is None:
            return

        use_cloning = self.cfg.use_cloning

        logger.info("Preparing stage for export (%d envs, cloning=%s)...", num_envs, use_cloning)
        create_cloning_attributes(stage, num_envs, use_cloning)

        export_path = "/tmp/stage_before_ovrtx.usda"
        export_stage_for_ovrtx(stage, export_path, num_envs, use_cloning)
        self._exported_usd_path = export_path
        logger.info("Exported to %s", export_path)

    def initialize(self, spec: CameraRenderSpec):
        """Initialize the OVRTX renderer with internal environment cloning.

        Args:
            spec: Tiled camera description (resolution, paths, data types).
        """
        width = spec.cfg.width
        height = spec.cfg.height
        num_envs = spec.num_instances
        data_types = spec.cfg.data_types if spec.cfg.data_types else ["rgb"]

        env_0_prefix = "/World/envs/env_0/"
        first_cam_path = spec.camera_prim_paths[0]
        if not first_cam_path.startswith(env_0_prefix):
            raise RuntimeError(f"Expected camera prim under '{env_0_prefix}', got '{first_cam_path}'")
        self._camera_rel_path = spec.camera_path_relative_to_env_0

        usd_scene_path = self._exported_usd_path
        use_cloning = self.cfg.use_cloning

        logger.info("Creating OVRTX renderer...")
        OVRTX_CONFIG = RendererConfig(
            log_file_path=self.cfg.log_file_path,
            log_level=self.cfg.log_level,
            read_gpu_transforms=_IS_OVRTX_0_3_0_OR_NEWER,
        )
        self._renderer = Renderer(OVRTX_CONFIG)
        assert self._renderer, "Renderer should be valid after creation"
        logger.info("OVRTX renderer created successfully")

        if usd_scene_path is not None:
            logger.info("Injecting camera definitions...")

            combined_usd_path, render_product_path = inject_cameras_into_usd(
                usd_scene_path,
                self.cfg,
                width=width,
                height=height,
                num_envs=num_envs,
                data_types=data_types,
                minimal_mode=_resolve_rtx_minimal_mode(data_types),
                camera_rel_path=self._camera_rel_path,
            )
            self._render_product_paths.append(render_product_path)

            logger.info("Loading USD into OvRTX...")
            try:
                handle = self._renderer.add_usd(combined_usd_path, path_prefix=None)
                self._usd_handles.append(handle)
                logger.info("USD loaded (path: %s, handle: %s)", combined_usd_path, handle)
            except Exception as e:
                logger.exception("Error loading USD: %s", e)
                raise

            if use_cloning and num_envs > 1:
                logger.info("Using OVRTX internal cloning")
                self._clone_environments_in_ovrtx(num_envs)
                self._update_scene_partitions_after_clone(combined_usd_path, num_envs)

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

    def _clone_environments_in_ovrtx(self, num_envs: int):
        """Clone base environment (env_0) to all other environments using OvRTX."""
        logger.info("Cloning base environment to %d targets...", num_envs - 1)
        source_path = "/World/envs/env_0"
        target_paths = [f"/World/envs/env_{i}" for i in range(1, num_envs)]
        try:
            self._renderer.clone_usd(source_path, target_paths)
            logger.info("Cloned %d environments successfully", len(target_paths))
        except Exception as e:
            logger.error("Failed to clone environments: %s", e)
            raise RuntimeError(f"OvRTX environment cloning failed: {e}")

    def _update_scene_partitions_after_clone(self, usd_file_path: str, num_envs: int):
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
            from isaaclab.sim import SimulationContext

            provider = SimulationContext.instance().initialize_scene_data_provider()
            newton_model = provider.get_newton_model()
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
                self._object_newton_indices = wp.array(newton_indices, dtype=wp.int32, device=DEVICE)
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
        if not self._initialized_scene:
            self.initialize(spec)
        return OVRTXRenderData(spec, DEVICE)

    # Map torch dtypes to their warp counterparts for zero-copy wrapping.
    _TORCH_TO_WP_DTYPE: dict[torch.dtype, Any] = {
        torch.uint8: wp.uint8,
        torch.float32: wp.float32,
        torch.int32: wp.int32,
    }

    def set_outputs(self, render_data: OVRTXRenderData, output_data: dict[str, torch.Tensor]) -> None:
        """Wrap caller-owned torch output tensors as zero-copy warp arrays.

        Aliased views over a contiguous sibling (e.g. ``rgb`` over ``rgba``) are
        skipped; any other non-contiguous tensor raises ``ValueError``.

        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`.
        """
        render_data.warp_buffers = {}
        for name, tensor in output_data.items():
            if not tensor.is_contiguous():
                if tensor.data_ptr() in {t.data_ptr() for t in output_data.values() if t.is_contiguous()}:
                    continue
                raise ValueError(
                    f"OVRTXRenderer.set_outputs: output '{name}' is non-contiguous and is not an"
                    " alias of a contiguous output tensor; cannot wrap as a zero-copy warp array."
                )
            wp_dtype = self._TORCH_TO_WP_DTYPE.get(tensor.dtype)
            if wp_dtype is None:
                raise ValueError(
                    f"OVRTXRenderer.set_outputs: unsupported torch dtype {tensor.dtype} for output"
                    f" '{name}'. Add it to OVRTXRenderer._TORCH_TO_WP_DTYPE."
                )
            torch_array = wp.from_torch(tensor)
            render_data.warp_buffers[name] = wp.array(
                ptr=torch_array.ptr,
                dtype=wp_dtype,
                shape=tuple(tensor.shape),
                device=torch_array.device,
                copy=False,
            )

    def update_transforms(self) -> None:
        """Sync physics objects to OVRTX."""
        if self._object_binding is None or self._object_newton_indices is None:
            return

        try:
            from isaaclab.sim import SimulationContext

            provider = SimulationContext.instance().initialize_scene_data_provider()
            newton_state = provider.get_newton_state()
            if newton_state is None:
                return
            body_q = getattr(newton_state, "body_q", None)
            if body_q is None:
                return

            with self._object_binding.map(device=Device.CUDA, device_id=0) as attr_mapping:
                ovrtx_transforms = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)
                wp.launch(
                    kernel=sync_newton_transforms_kernel,
                    dim=len(self._object_newton_indices),
                    inputs=[ovrtx_transforms, self._object_newton_indices, body_q],
                    device=DEVICE,
                )
        except Exception as e:
            logger.warning("Failed to update object transforms: %s", e)

    def update_camera(
        self,
        render_data: OVRTXRenderData,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> None:
        """Update camera transforms in OVRTX binding."""
        num_envs = positions.shape[0]
        camera_quats_opengl = convert_camera_frame_orientation_convention(orientations, origin="world", target="opengl")
        camera_positions_wp = wp.from_torch(positions.contiguous(), dtype=wp.vec3)
        camera_orientations_wp = wp.from_torch(camera_quats_opengl.contiguous(), dtype=wp.quatf)
        camera_transforms = wp.zeros(num_envs, dtype=wp.mat44d, device=DEVICE)
        wp.launch(
            kernel=create_camera_transforms_kernel,
            dim=num_envs,
            inputs=[camera_positions_wp, camera_orientations_wp, camera_transforms],
            device=DEVICE,
        )
        if self._camera_binding is not None:
            with self._camera_binding.map(device=Device.CUDA, device_id=0) as attr_mapping:
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
            self._output_semantic_color_buffer = wp.zeros(shape=input_ids.shape, dtype=wp.uint32, device=DEVICE)

        output_colors = self._output_semantic_color_buffer

        wp.launch(
            kernel=(
                generate_random_colors_from_ids_kernel
                if _IS_OVRTX_0_3_0_OR_NEWER
                else generate_random_colors_from_ids_kernel_legacy
            ),
            dim=input_ids.shape,
            inputs=[input_ids, output_colors],
            device=DEVICE,
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
            device=DEVICE,
        )

    def _extract_depth_tiles(
        self, render_data: OVRTXRenderData, tiled_depth_data: wp.array, output_buffers: dict
    ) -> None:
        """Extract per-env depth tiles into output_buffers (single kernel launch)."""
        kernel = extract_all_depth_tiles_kernel if _IS_OVRTX_0_3_0_OR_NEWER else extract_all_depth_tiles_kernel_legacy

        for depth_type in ["depth", "distance_to_image_plane", "distance_to_camera"]:
            if depth_type in output_buffers:
                wp.launch(
                    kernel=kernel,
                    dim=(render_data.num_envs, render_data.height, render_data.width),
                    inputs=[
                        tiled_depth_data,
                        output_buffers[depth_type],
                        render_data.num_cols,
                        render_data.width,
                        render_data.height,
                    ],
                    device=DEVICE,
                )

    def _process_render_frame(self, render_data: OVRTXRenderData, frame, output_buffers: dict) -> None:
        """Extract RGB, depth, albedo, and semantic from a single render frame into output_buffers."""
        if "LdrColor" in frame.render_vars:
            buffer_key = None

            if "rgba" in output_buffers:
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
            if self._usd_handles:
                for handle in self._usd_handles:
                    try:
                        self._renderer.remove_usd(handle)
                    except Exception as e:
                        logger.warning("Error removing USD: %s", e)
                self._usd_handles.clear()
            self._renderer = None

        self._render_product_paths.clear()
        self._output_semantic_color_buffer = None
        self._initialized_scene = False
