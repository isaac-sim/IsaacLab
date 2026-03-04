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

import math
import os
import weakref
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

# The ovrtx C library links to its own version of the USD libraries. Having
# the pxr Python package available can cause the C library to load an
# incompatible version of libusd, potentially leading to undefined behavior.
# By setting OVRTX_SKIP_USD_CHECK, we prevent the C library from loading the pxr Python package.
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

from ovrtx import Device, PrimMode, Renderer, RendererConfig, Semantic

from isaaclab.renderers.base_renderer import BaseRenderer
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from .ovrtx_renderer_cfg import OVRTXRendererCfg
from .ovrtx_renderer_kernels import (
    DEVICE,
    create_camera_transforms_kernel,
    extract_depth_tile_from_tiled_buffer_kernel,
    extract_tile_from_tiled_buffer_kernel,
    sync_newton_transforms_kernel,
)
from .ovrtx_usd import (
    create_cloning_attributes,
    export_stage_for_ovrtx,
    inject_cameras_into_usd,
)

if TYPE_CHECKING:
    from isaaclab.sensors import SensorBase


class OVRTXRenderData:
    """OVRTX-specific RenderData. Holds warp output buffers and weak ref to sensor.

    Follows Newton Warp pattern: weak ref to sensor avoids circular reference while
    allowing access to sensor config when needed.
    """

    @staticmethod
    def _create_warp_buffers(
        width: int,
        height: int,
        num_envs: int,
        data_types: list[str],
        device,
    ) -> dict:
        """Create warp output buffers for OVRTX renderer."""
        buffers = {}
        if any(dt in ("rgba", "rgb") for dt in data_types):
            buffers["rgba"] = wp.zeros((num_envs, height, width, 4), dtype=wp.uint8, device=device)
            buffers["rgb"] = buffers["rgba"][:, :, :, :3]
        if "albedo" in data_types:
            buffers["albedo"] = wp.zeros((num_envs, height, width, 4), dtype=wp.uint8, device=device)
        if "semantic_segmentation" in data_types:
            buffers["semantic_segmentation"] = wp.zeros((num_envs, height, width, 4), dtype=wp.uint8, device=device)
        for depth_key in ("depth", "distance_to_image_plane", "distance_to_camera"):
            if depth_key in data_types:
                buffers[depth_key] = wp.zeros((num_envs, height, width, 1), dtype=wp.float32, device=device)
        return buffers

    def __init__(self, sensor: SensorBase, device):
        """Create render data from sensor. Holds weak ref to avoid circular reference."""
        self.sensor: weakref.ref[object] | None = weakref.ref(sensor)
        self.width = sensor.cfg.width
        self.height = sensor.cfg.height
        self.num_envs = sensor._num_envs
        self.data_types = sensor.cfg.data_types if sensor.cfg.data_types else ["rgb"]
        self.num_cols = math.ceil(math.sqrt(self.num_envs))
        self.num_rows = math.ceil(self.num_envs / self.num_cols)
        self.warp_buffers = self._create_warp_buffers(self.width, self.height, self.num_envs, self.data_types, device)


class OVRTXRenderer(BaseRenderer):
    """OVRTX Renderer implementation using the ovrtx library.

    This renderer uses the ovrtx library for high-fidelity RTX-based rendering,
    providing ray-traced rendering capabilities for Isaac Lab environments.
    """

    cfg: OVRTXRendererCfg

    def __init__(self, cfg: OVRTXRendererCfg):
        self.cfg = cfg
        self._usd_handles = []
        self._render_product_paths = []
        self._camera_binding = None
        self._object_binding = None
        self._object_newton_indices: wp.array | None = None
        self._initialized_scene = False
        self._sensor_ref: weakref.ref[object] | None = None

    def initialize(self, sensor: SensorBase):
        """Initialize the OVRTX renderer with internal environment cloning.

        Only env_0 is exported to USD; OVRTX clone_usd() replicates environments
        for fast initialization (O(1) or O(log N) for many envs).

        Args:
            sensor: The TiledCamera sensor. width, height, num_envs, data_types are
                obtained from sensor when needed. Weak ref stored to avoid circular ref.
        """
        self._sensor_ref = weakref.ref(sensor)
        width = sensor.cfg.width
        height = sensor.cfg.height
        num_envs = sensor._num_envs
        data_types = sensor.cfg.data_types if sensor.cfg.data_types else ["rgb"]

        camera_prim_path = sensor.cfg.prim_path
        camera_prim_name = (camera_prim_path or "").strip().split("/")[-1] or "Camera"
        stage = sensor.stage
        usd_scene_path = None
        use_cloning = self.cfg.use_cloning
        if stage is not None:
            print(f"[OVRTX] Preparing stage for export ({num_envs} envs, cloning={use_cloning})...")
            create_cloning_attributes(stage, camera_prim_name, num_envs, use_cloning)
            export_path = "/tmp/stage_before_ovrtx.usda"
            export_stage_for_ovrtx(stage, export_path, num_envs, use_cloning)
            usd_scene_path = export_path
            print(f"   ✓ Exported to {export_path}")

        print("Creating OVRTX renderer...")
        OVRTX_CONFIG = RendererConfig(
            log_file_path=self.cfg.log_file_path,
            log_level=self.cfg.log_level,
            read_gpu_transforms=False,
        )
        self._renderer = Renderer(OVRTX_CONFIG)
        assert self._renderer, "Renderer should be valid after creation"
        print("OVRTX renderer created successfully!")

        if usd_scene_path is not None:
            print("[OVRTX] Injecting camera definitions...")
            combined_usd_path, render_product_path = inject_cameras_into_usd(
                usd_scene_path,
                self.cfg,
                width=width,
                height=height,
                num_envs=num_envs,
                data_types=data_types,
            )
            self._render_product_paths.append(render_product_path)

            print("[OVRTX] Loading USD into OvRTX...")
            try:
                handle = self._renderer.add_usd(combined_usd_path, path_prefix=None)
                self._usd_handles.append(handle)
                print(f"USD loaded (path: {combined_usd_path}, handle: {handle})")
            except Exception as e:
                print(f"ERROR loading USD: {e}")
                import traceback

                traceback.print_exc()
                raise

            if use_cloning and num_envs > 1:
                print("[OVRTX] Using OVRTX internal cloning")
                self._clone_environments_in_ovrtx(num_envs)
                self._update_scene_partitions_after_clone(combined_usd_path, num_envs)

            self._initialized_scene = True

            camera_paths = [f"/World/envs/env_{i}/Camera" for i in range(num_envs)]
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
                print(f"  ⚠ Warning: Failed to write omni:resetXformStack: {e}")

            if self._camera_binding is not None:
                print("  ✓ Camera binding created successfully")
            else:
                print("  ✗ WARNING: Camera binding is None!")

            self._setup_object_bindings()

    def _clone_environments_in_ovrtx(self, num_envs: int):
        """Clone base environment (env_0) to all other environments using OvRTX."""
        print(f"[OVRTX OPTIMIZE] Cloning base environment to {num_envs - 1} targets...")
        source_path = "/World/envs/env_0"
        target_paths = [f"/World/envs/env_{i}" for i in range(1, num_envs)]
        try:
            self._renderer.clone_usd(source_path, target_paths)
            print(f"  ✓ Cloned {len(target_paths)} environments successfully")
        except Exception as e:
            print(f"  ✗ ERROR: Failed to clone environments: {e}")
            raise RuntimeError(f"OvRTX environment cloning failed: {e}")

    def _update_scene_partitions_after_clone(self, usd_file_path: str, num_envs: int):
        """Update scene partition attributes on cloned environments and cameras in OvRTX."""
        print(f"[OVRTX] Writing scene partitions for {num_envs} environments...")
        partition_tokens = [f"env_{i}" for i in range(num_envs)]
        env_prim_paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
        camera_prim_paths = [f"/World/envs/env_{i}/Camera" for i in range(num_envs)]

        try:
            self._renderer.write_attribute(
                env_prim_paths,
                "primvars:omni:scenePartition",
                partition_tokens,
                semantic=Semantic.TOKEN_STRING,
            )
            print(f"  ✓ Written primvars:omni:scenePartition to {num_envs} environments")

            self._renderer.write_attribute(
                camera_prim_paths,
                "omni:scenePartition",
                partition_tokens,
                semantic=Semantic.TOKEN_STRING,
            )
            print(f"  ✓ Written omni:scenePartition to {num_envs} cameras")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to write scene partitions: {e}")
            import traceback

            traceback.print_exc()

    def _setup_object_bindings(self):
        """Setup OVRTX bindings for scene objects to sync with Newton physics."""
        try:
            from isaaclab.sim import SimulationContext
            from isaaclab.visualizers import VisualizerCfg

            provider = SimulationContext.instance().initialize_scene_data_provider(
                [VisualizerCfg(visualizer_type="newton")]
            )
            newton_model = provider.get_newton_model()
            if newton_model is None:
                print("[OVRTX] Newton model not available, skipping object bindings")
                return

            all_body_paths = getattr(newton_model, "body_label", None)
            if all_body_paths is None:
                print("[OVRTX] Newton model has no body_label, skipping object bindings")
                return

            object_paths = []
            newton_indices = []
            for idx, path in enumerate(all_body_paths):
                if "/World/envs/" in path and "Camera" not in path and "GroundPlane" not in path:
                    object_paths.append(path)
                    newton_indices.append(idx)

            if len(object_paths) == 0:
                print("[OVRTX] No dynamic objects found for binding")
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
                print(f"  ⚠ Warning: Failed to write omni:resetXformStack on objects: {e}")

            if self._object_binding is not None:
                print("  ✓ Object binding created successfully")
                self._object_newton_indices = wp.array(newton_indices, dtype=wp.int32, device=DEVICE)
            else:
                print("  ✗ WARNING: Object binding is None!")
        except ImportError:
            print("[OVRTX] Newton not available, skipping object bindings")
        except Exception as e:
            print(f"[OVRTX] Error setting up object bindings: {e}")

    def create_render_data(self, sensor: SensorBase) -> OVRTXRenderData:
        """Create OVRTX-specific RenderData with GPU buffers.

        Performs OVRTX initialization (stage export, USD load, bindings) on first call,
        matching the interface of Isaac RTX and Newton Warp which need no separate initialize().
        RenderData holds weak ref to sensor (Newton pattern) to avoid circular reference.
        """
        if not self._initialized_scene:
            self.initialize(sensor)
        return OVRTXRenderData(sensor, DEVICE)

    def set_outputs(self, render_data: OVRTXRenderData, output_data: dict) -> None:
        """No-op; OVRTX uses internal warp buffers."""
        pass

    def update_transforms(self) -> None:
        """Sync physics objects to OVRTX."""
        if self._object_binding is None or self._object_newton_indices is None:
            return

        try:
            from isaaclab.sim import SimulationContext
            from isaaclab.visualizers import VisualizerCfg

            provider = SimulationContext.instance().initialize_scene_data_provider(
                [VisualizerCfg(visualizer_type="newton")]
            )
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
            print(f"[OVRTX] Warning: Failed to update object transforms: {e}")

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

    def write_output(
        self,
        render_data: OVRTXRenderData,
        output_name: str,
        output_data: torch.Tensor,
    ) -> None:
        """Copy from render_data warp buffer to output tensor."""
        if output_name not in render_data.warp_buffers:
            return
        src = render_data.warp_buffers[output_name]
        if src.ptr != output_data.data_ptr():
            wp.copy(dest=wp.from_torch(output_data), src=src)

    def _extract_rgba_tiles(
        self,
        render_data: OVRTXRenderData,
        tiled_data: wp.array,
        output_buffers: dict,
        buffer_key: str,
        suffix: str = "",
    ) -> None:
        """Extract per-env RGBA tiles from tiled buffer into output_buffers."""
        for env_idx in range(render_data.num_envs):
            tile_x = env_idx % render_data.num_cols
            tile_y = env_idx // render_data.num_cols
            wp.launch(
                kernel=extract_tile_from_tiled_buffer_kernel,
                dim=(render_data.height, render_data.width),
                inputs=[
                    tiled_data,
                    output_buffers[buffer_key][env_idx],
                    tile_x,
                    tile_y,
                    render_data.width,
                    render_data.height,
                ],
                device=DEVICE,
            )

    def _extract_depth_tiles(
        self, render_data: OVRTXRenderData, tiled_depth_data: wp.array, output_buffers: dict
    ) -> None:
        """Extract per-env depth tiles into output_buffers."""
        for env_idx in range(render_data.num_envs):
            tile_x = env_idx % render_data.num_cols
            tile_y = env_idx // render_data.num_cols
            for depth_type in ["depth", "distance_to_image_plane", "distance_to_camera"]:
                if depth_type in output_buffers:
                    wp.launch(
                        kernel=extract_depth_tile_from_tiled_buffer_kernel,
                        dim=(render_data.height, render_data.width),
                        inputs=[
                            tiled_depth_data,
                            output_buffers[depth_type][env_idx],
                            tile_x,
                            tile_y,
                            render_data.width,
                            render_data.height,
                        ],
                        device=DEVICE,
                    )

    def _process_render_frame(self, render_data: OVRTXRenderData, frame, output_buffers: dict) -> None:
        """Extract RGB, depth, albedo, and semantic from a single render frame into output_buffers."""
        rgb_render_var = (
            "SimpleShadingSD"
            if "SimpleShadingSD" in frame.render_vars
            else "LdrColor"
            if "LdrColor" in frame.render_vars
            else None
        )
        if rgb_render_var and "rgba" in output_buffers:
            with frame.render_vars[rgb_render_var].map(device=Device.CUDA) as mapping:
                tiled_data = wp.from_dlpack(mapping.tensor)
                self._extract_rgba_tiles(render_data, tiled_data, output_buffers, "rgba", suffix="rgb")

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

        if "SemanticSegmentationSD" in frame.render_vars and "semantic_segmentation" in output_buffers:
            with frame.render_vars["SemanticSegmentationSD"].map(device=Device.CUDA) as mapping:
                tiled_semantic_data = wp.from_dlpack(mapping.tensor)
                if tiled_semantic_data.dtype == wp.uint32:
                    semantic_torch = wp.to_torch(tiled_semantic_data)
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
            print(f"Warning: OVRTX rendering failed: {e}")
            import traceback

            traceback.print_exc()

    def cleanup(self, render_data: OVRTXRenderData | None) -> None:
        """Release renderer resources. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""
        if render_data is not None:
            render_data.sensor = None  # Break weak ref (Newton pattern)
        self._sensor_ref = None

        # Unbind before tearing down renderer
        def _safe_unbind(binding, name: str) -> None:
            if binding is None:
                return
            try:
                binding.unbind()
            except Exception as e:
                if "destroyed" not in str(e).lower():
                    print(f"Warning: Error unbinding {name}: {e}")

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
                        print(f"Warning: Error removing USD: {e}")
                self._usd_handles.clear()
            self._renderer = None

        self._render_product_paths.clear()
        self._initialized_scene = False
