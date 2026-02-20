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

- **ovrtx_renderer_kernels.py**: Warp GPU kernels and DEVICE constant. Provides
  create_camera_transforms_kernel, extract_tile_from_tiled_buffer_kernel,
  extract_depth_tile_from_tiled_buffer_kernel, sync_newton_transforms_kernel, and
  normalize_depth_to_uint8(). No OVRTX or renderer state.

- **ovrtx_usd.py**: USD helpers for OVRTX: render var config, Render scope string building,
  inject_cameras_into_usd (read USD, append Render scope, write temp file), and
  deactivate_cloned_envs / reactivate_prims for stage prim visibility.
"""

import math
import os
from dataclasses import MISSING

import numpy as np
import torch
import warp as wp

# Set environment variables for OVRTX
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

from ovrtx import Device, PrimMode, Renderer, RendererConfig, Semantic

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
    export_stage_for_ovrtx,
    inject_cameras_into_usd,
    set_scene_partition_attributes,
)
from .renderer import RendererBase


class OVRTXRenderer(RendererBase):
    """OVRTX Renderer implementation using the ovrtx library.

    This renderer uses the ovrtx library for high-fidelity RTX-based rendering,
    providing ray-traced rendering capabilities for Isaac Lab environments.
    """

    _renderer: Renderer | None = None
    _usd_handles: list | None = None
    _camera_binding = None
    _object_binding = None  # Binding for scene objects (robot, manipulated objects, etc.)
    _object_newton_indices: wp.array | None = None  # Newton body indices for objects
    _render_product_paths: list[str] = []
    _initialized_scene = False
    _frame_counter: int = 0  # Track frame number for image filenames

    def __init__(self, cfg: OVRTXRendererCfg):
        super().__init__(cfg)
        self._usd_handles = []
        self._render_product_paths = []
        self._frame_counter = 0
        
        # Calculate tiled dimensions properly (not a square grid)
        # Use same logic as TiledCamera._tiling_grid_shape()
        self._num_cols = math.ceil(math.sqrt(self._num_envs))
        self._num_rows = math.ceil(self._num_envs / self._num_cols)
        self._tiled_width = self._num_cols * self._width
        self._tiled_height = self._num_rows * self._height
        
        # Store data types from config (handle MISSING from configclass)
        dt = getattr(cfg, "data_types", MISSING)
        self._data_types = dt if (dt is not MISSING and dt) else ["rgb"]

        self._simple_shading_mode = getattr(cfg, "simple_shading_mode", True)

    def _clone_environments_in_ovrtx(self):
        """Clone base environment (env_0) to all other environments using OvRTX.
        
        This uses OvRTX's efficient clone_usd() method to replicate the base environment,
        which is much faster than loading all N environments from USD.
        
        Performance: O(1) or O(log N) vs O(N) for traditional loading.
        """
        print(f"[OVRTX OPTIMIZE] Cloning base environment to {self._num_envs - 1} targets...")
        
        # Source: base environment
        source_path = "/World/envs/env_0"
        
        # Targets: all cloned environments
        target_paths = [f"/World/envs/env_{i}" for i in range(1, self._num_envs)]
        
        # Clone using OvRTX API
        try:
            self._renderer.clone_usd(source_path, target_paths)
            print(f"  ✓ Cloned {len(target_paths)} environments successfully")
        except Exception as e:
            print(f"  ✗ ERROR: Failed to clone environments: {e}")
            raise RuntimeError(f"OvRTX environment cloning failed: {e}")

    def _update_scene_partitions_after_clone(self, usd_file_path: str):
        """Update scene partition attributes on cloned environments and cameras in OvRTX.
        
        After OvRTX clones environments, we need to update the partition attributes
        so each environment and camera has the correct partition identifier.
        
        This uses OvRTX's write_attribute API to directly write token attributes,
        similar to the C++ implementation.
        
        Args:
            usd_file_path: Path to the USD file (not used, kept for compatibility)
        """
        print(f"[OVRTX] Writing scene partitions for {self._num_envs} environments...")
        
        # Build partition token strings: "env_0", "env_1", ..., "env_N-1"
        partition_tokens = [f"env_{i}" for i in range(self._num_envs)]
        
        # Build environment prim paths
        env_prim_paths = [f"/World/envs/env_{i}" for i in range(self._num_envs)]
        
        # Build camera prim paths
        camera_prim_paths = [f"/World/envs/env_{i}/Camera" for i in range(self._num_envs)]
        
        try:
            # Write primvars:omni:scenePartition to environment prims
            env_partition_binding = self._renderer.bind_attribute(
                prim_paths=env_prim_paths,
                attribute_name="primvars:omni:scenePartition",
                semantic=Semantic.TOKEN_STRING,
                prim_mode=PrimMode.EXISTING_ONLY,
            )
            
            if env_partition_binding is not None:
                # Use numpy for token strings
                partition_array = np.array(partition_tokens, dtype="U32")
                self._renderer.write_attribute(env_partition_binding, tensor=partition_array)
                print(f"  ✓ Written primvars:omni:scenePartition to {self._num_envs} environments")
            else:
                print(f"  ⚠ Warning: Failed to bind primvars:omni:scenePartition on environments")
            
            # Write omni:scenePartition to camera prims
            cam_partition_binding = self._renderer.bind_attribute(
                prim_paths=camera_prim_paths,
                attribute_name="omni:scenePartition",
                semantic=Semantic.TOKEN_STRING,
                prim_mode=PrimMode.EXISTING_ONLY,
            )
            
            if cam_partition_binding is not None:
                # Reuse the same partition tokens for cameras (numpy, Warp has no string arrays)
                cam_partition_array = np.array(partition_tokens, dtype="U32")
                self._renderer.write_attribute(cam_partition_binding, tensor=cam_partition_array)
                print(f"  ✓ Written omni:scenePartition to {self._num_envs} cameras")
            else:
                print(f"  ⚠ Warning: Failed to bind omni:scenePartition on cameras")
                
        except Exception as e:
            print(f"  ⚠ Warning: Failed to write scene partitions: {e}")
            import traceback
            traceback.print_exc()


    def initialize(self, stage=None, camera_prim_path=None):
        """Initialize the OVRTX renderer with internal environment cloning.

        Only env_0 is exported to USD; OVRTX clone_usd() replicates environments
        for fast initialization (O(1) or O(log N) for many envs).

        Args:
            stage: Optional USD stage. If provided, prepared and exported for OVRTX.
            camera_prim_path: Full path pattern for the camera (e.g. "/World/envs/env_.*/Camera").
                The camera name under each env is derived from the last path segment.
        """
        camera_prim_name = (camera_prim_path or "").strip().split("/")[-1] or "Camera"
        usd_scene_path = None
        # If stage provided, prepare and export it (partition attributes + optional deactivate for cloning)
        if stage is not None:
            print(f"[OVRTX] Preparing stage for export ({self._num_envs} envs, camera_prim={camera_prim_name!r})...")
            total_objs = set_scene_partition_attributes(stage, self._num_envs, camera_prim_name)
            print(f"   ✓ Set scene partition attributes on {self._num_envs} envs, {total_objs} objects")
            export_path = "/tmp/stage_before_ovrtx.usda"
            export_stage_for_ovrtx(stage, export_path, self._num_envs)
            usd_scene_path = export_path
            print(f"   ✓ Exported to {export_path}")

        # Log USD version info for debugging
        from pxr import Usd
        print(f"[OVRTX] USD Version: {Usd.GetVersion()}")
        print(f"[OVRTX] USD Module: {Usd.__file__}")
        
        print("Creating OVRTX renderer...")
        
        # Add simple shading mode configuration if enabled
        if self._simple_shading_mode:
            print(f"[OVRTX] Simple shading mode ENABLED")
        else:
            print(f"[OVRTX] Simple shading mode DISABLED (using full RTX path tracing)")
        
        # Create renderer config with proper parameters
        OVRTX_CONFIG = RendererConfig(
            log_file_path="/tmp/ovrtx_renderer.log",
            log_level="warning",
        )
        self._renderer = Renderer(OVRTX_CONFIG)
        assert self._renderer, "Renderer should be valid after creation"
        print("OVRTX renderer created successfully!")

        # Initialize output buffers
        self._initialize_output()
        
        # If a USD scene is provided, load and optionally clone
        if usd_scene_path is not None:
            from pxr import Usd
            
            # Load USD file into OvRTX
            # Export contains only env_0 when num_envs > 1; OVRTX clones after load
            print(f"[OVRTX] Injecting camera definitions...")
            combined_usd_path, render_product_path = inject_cameras_into_usd(
                usd_scene_path,
                self._num_envs,
                self._tiled_width,
                self._tiled_height,
                self._data_types,
                self._simple_shading_mode,
            )
            self._render_product_paths.append(render_product_path)
            
            print(f"[OVRTX] Loading USD into OvRTX...")
            try:
                handle = self._renderer.add_usd(combined_usd_path, path_prefix=None)
                self._usd_handles.append(handle)
                print(f"   ✓ USD loaded (handle: {handle})")
            except Exception as e:
                print(f"   ✗ ERROR loading USD: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Clone base environment to all other environments in OvRTX when num_envs > 1
            if self._num_envs > 1:
                print(f"[OVRTX] Using OVRTX internal cloning")
                self._clone_environments_in_ovrtx()
                self._update_scene_partitions_after_clone(combined_usd_path)

            self._initialized_scene = True
            
            # Create binding for camera transforms (all environments now exist in OVRTX)
            camera_paths = [f"/World/envs/env_{i}/Camera" for i in range(self._num_envs)]
            self._camera_binding = self._renderer.bind_attribute(
                prim_paths=camera_paths,
                attribute_name="omni:xform",
                semantic=Semantic.XFORM_MAT4x4,
                prim_mode=PrimMode.EXISTING_ONLY,
            )
            
            if self._camera_binding is not None:
                print(f"  ✓ Camera binding created successfully")
            else:
                print(f"  ✗ WARNING: Camera binding is None!")
            
            # Setup object bindings for Newton physics sync
            self._setup_object_bindings()
        else:
            pass  # No USD scene: cameras as root layer not implemented

    def _setup_object_bindings(self):
        """Setup OVRTX bindings for scene objects to sync with Newton physics.
        
        This creates bindings for all dynamic objects (robot bodies, manipulated objects)
        that need to be updated each frame from Newton's physics state.
        """
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager
            
            newton_model = NewtonManager.get_model()
            if newton_model is None:
                print("[OVRTX] Newton model not available, skipping object bindings")
                return
            
            # Get all body paths from Newton
            # Filter out static objects (plane, lights) and cameras
            all_body_paths = newton_model.body_key
            
            # Filter to only dynamic objects in envs
            # Typically: /World/envs/env_X/Robot/..., /World/envs/env_X/object, etc.
            object_paths = []
            newton_indices = []
            
            for idx, path in enumerate(all_body_paths):
                # Include objects in /World/envs/ but exclude cameras and ground plane
                if "/World/envs/" in path and "Camera" not in path and "GroundPlane" not in path:
                    object_paths.append(path)
                    newton_indices.append(idx)
            
            if len(object_paths) == 0:
                print("[OVRTX] No dynamic objects found for binding")
                return

            # Create OVRTX binding for all objects at once
            self._object_binding = self._renderer.bind_attribute(
                prim_paths=object_paths,
                attribute_name="omni:xform",
                semantic=Semantic.XFORM_MAT4x4,
                prim_mode=PrimMode.EXISTING_ONLY,
            )
            
            if self._object_binding is not None:
                print(f"  ✓ Object binding created successfully")
                # Store Newton body indices for later lookup
                self._object_newton_indices = wp.array(newton_indices, dtype=wp.int32, device=DEVICE)
            else:
                print(f"  ✗ WARNING: Object binding is None!")
                
        except ImportError:
            print("[OVRTX] Newton not available, skipping object bindings")
        except Exception as e:
            print(f"[OVRTX] Error setting up object bindings: {e}")
    
    def add_usd_scene(self, usd_file_path: str, path_prefix: str | None = None):
        """Add a USD scene file to the renderer.
        
        This allows loading geometry and scene content into the renderer.
        
        Args:
            usd_file_path: Path to the USD file to load
            path_prefix: Optional path prefix for the USD content
            
        Returns:
            USD handle that can be used to remove the scene later
        """
        if self._renderer is None:
            raise RuntimeError("Renderer not initialized. Call initialize() first.")
        
        print(f"Loading USD scene: {usd_file_path}")
        handle = self._renderer.add_usd(usd_file_path, path_prefix)
        
        if self._usd_handles is not None:
            self._usd_handles.append(handle)
        
        print(f"USD scene loaded successfully (handle: {handle})")
        return handle

    def _initialize_output(self):
        """Initialize the output of the renderer."""
        # Create output buffers based on requested data types
        
        # RGBA/RGB buffers (shared)
        if any(dt in ["rgba", "rgb"] for dt in self._data_types):
            self._output_data_buffers["rgba"] = wp.zeros(
                (self._num_envs, self._height, self._width, 4), dtype=wp.uint8, device=DEVICE
            )
            # Create RGB view that references the same underlying array as RGBA, but only first 3 channels
            self._output_data_buffers["rgb"] = self._output_data_buffers["rgba"][:, :, :, :3]
        
        # Albedo buffer (4-channel RGBA format, similar to rgb/rgba)
        if "albedo" in self._data_types:
            self._output_data_buffers["albedo"] = wp.zeros(
                (self._num_envs, self._height, self._width, 4), dtype=wp.uint8, device=DEVICE
            )
        
        # Semantic segmentation buffer (4-channel RGBA format for colorized output)
        if "semantic_segmentation" in self._data_types:
            self._output_data_buffers["semantic_segmentation"] = wp.zeros(
                (self._num_envs, self._height, self._width, 4), dtype=wp.uint8, device=DEVICE
            )
        
        # Depth buffers (note: "depth" is an alias for "distance_to_image_plane")
        if "depth" in self._data_types:
            self._output_data_buffers["depth"] = wp.zeros(
                (self._num_envs, self._height, self._width, 1), dtype=wp.float32, device=DEVICE
            )
        
        if "distance_to_image_plane" in self._data_types:
            self._output_data_buffers["distance_to_image_plane"] = wp.zeros(
                (self._num_envs, self._height, self._width, 1), dtype=wp.float32, device=DEVICE
            )
        
        if "distance_to_camera" in self._data_types:
            self._output_data_buffers["distance_to_camera"] = wp.zeros(
                (self._num_envs, self._height, self._width, 1), dtype=wp.float32, device=DEVICE
            )

    def _extract_rgba_tiles(
        self, tiled_data: wp.array, buffer_key: str, suffix: str = ""
    ) -> None:
        """Extract per-env RGBA tiles from tiled buffer and optionally save to disk."""
        for env_idx in range(self._num_envs):
            tile_x = env_idx % self._num_cols
            tile_y = env_idx // self._num_cols
            wp.launch(
                kernel=extract_tile_from_tiled_buffer_kernel,
                dim=(self._height, self._width),
                inputs=[
                    tiled_data,
                    self._output_data_buffers[buffer_key][env_idx],
                    tile_x,
                    tile_y,
                    self._width,
                    self._height,
                ],
                device=DEVICE,
            )

    def _extract_depth_tiles(self, tiled_depth_data: wp.array) -> None:
        """Extract per-env depth tiles and populate all depth-type buffers; save depth images."""
        for env_idx in range(self._num_envs):
            tile_x = env_idx % self._num_cols
            tile_y = env_idx // self._num_cols
            for depth_type in ["depth", "distance_to_image_plane", "distance_to_camera"]:
                if depth_type in self._output_data_buffers:
                    wp.launch(
                        kernel=extract_depth_tile_from_tiled_buffer_kernel,
                        dim=(self._height, self._width),
                        inputs=[
                            tiled_depth_data,
                            self._output_data_buffers[depth_type][env_idx],
                            tile_x,
                            tile_y,
                            self._width,
                            self._height,
                        ],
                        device=DEVICE,
                    )

    def _process_render_frame(self, frame) -> None:
        """Extract RGB, depth, albedo, and semantic from a single render frame into buffers."""
        # RGB/RGBA
        rgb_render_var = "SimpleShadingSD" if "SimpleShadingSD" in frame.render_vars else "LdrColor" if "LdrColor" in frame.render_vars else None
        if rgb_render_var and "rgba" in self._output_data_buffers:
            with frame.render_vars[rgb_render_var].map(device=Device.CUDA) as mapping:
                tiled_data = wp.from_dlpack(mapping.tensor)
                self._extract_rgba_tiles(tiled_data, "rgba", suffix="rgb")

        # Depth
        for depth_var in ["DistanceToImagePlaneSD", "DepthSD"]:
            if depth_var not in frame.render_vars:
                continue
            with frame.render_vars[depth_var].map(device=Device.CUDA) as mapping:
                tiled_depth_data = wp.from_dlpack(mapping.tensor)
                if tiled_depth_data.dtype == wp.uint32:
                    tiled_depth_data = wp.from_torch(
                        wp.to_torch(tiled_depth_data).view(torch.float32), dtype=wp.float32
                    )
                self._extract_depth_tiles(tiled_depth_data)
            break

        # Albedo
        if "DiffuseAlbedoSD" in frame.render_vars and "albedo" in self._output_data_buffers:
            with frame.render_vars["DiffuseAlbedoSD"].map(device=Device.CUDA) as mapping:
                tiled_albedo_data = wp.from_dlpack(mapping.tensor)
                self._extract_rgba_tiles(tiled_albedo_data, "albedo", suffix="albedo")

        # Semantic segmentation
        if "SemanticSegmentationSD" in frame.render_vars and "semantic_segmentation" in self._output_data_buffers:
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
                    tiled_semantic_data, "semantic_segmentation", suffix="semantic"
                )

    def render(self, camera_positions: torch.Tensor, camera_orientations: torch.Tensor, intrinsic_matrices: torch.Tensor):
        """Render the scene using OVRTX.
        
        Args:
            camera_positions: Tensor of shape (num_envs, 3) - camera positions in world frame
            camera_orientations: Tensor of shape (num_envs, 4) - camera quaternions (x, y, z, w) in world frame
            intrinsic_matrices: Tensor of shape (num_envs, 3, 3) - camera intrinsic matrices
        """
        # Scene should already be set up during initialize()
        if not self._initialized_scene:
            raise RuntimeError("Scene not initialized. This should not happen - scene setup should occur in initialize()")
        
        # Increment frame counter
        self._frame_counter += 1
        
        num_envs = camera_positions.shape[0]
        
        # Convert camera orientations from world convention to OpenGL convention
        # This is necessary because Camera._update_poses() converts from opengl to world,
        # but OVRTX expects OpenGL convention (same as Newton Warp Renderer)
        camera_quats_opengl = convert_camera_frame_orientation_convention(
            camera_orientations, origin="world", target="opengl"
        )
        
        # Convert torch tensors to warp arrays
        camera_positions_wp = wp.from_torch(camera_positions.contiguous(), dtype=wp.vec3)
        camera_orientations_wp = wp.from_torch(camera_quats_opengl.contiguous(), dtype=wp.quatf)
        
        # Create camera transforms array
        camera_transforms = wp.zeros(num_envs, dtype=wp.mat44d, device=DEVICE)
        
        # Launch kernel to populate transforms
        wp.launch(
            kernel=create_camera_transforms_kernel,
            dim=num_envs,
            inputs=[camera_positions_wp, camera_orientations_wp, camera_transforms],
            device=DEVICE,
        )
        
        # Update camera transforms in the scene using the binding
        if self._camera_binding is not None:
            with self._camera_binding.map(device=Device.CUDA, device_id=0) as attr_mapping:
                wp_transforms_view = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)
                wp.copy(wp_transforms_view, camera_transforms)
                # Unmap will commit the changes
        
        # Update object transforms from Newton physics
        self._update_object_transforms()
        
        # Step the renderer and process the tiled frame
        if self._renderer is not None and len(self._render_product_paths) > 0:
            try:
                products = self._renderer.step(
                    render_products=set(self._render_product_paths),
                    delta_time=1.0 / 60.0,
                )
                product_path = self._render_product_paths[0]
                if product_path in products and len(products[product_path].frames) > 0:
                    self._process_render_frame(products[product_path].frames[0])
            except Exception as e:
                print(f"Warning: OVRTX rendering failed: {e}")
                import traceback
                traceback.print_exc()

    def _update_object_transforms(self):
        """Update object transforms from Newton physics state to OVRTX.
        
        Syncs all dynamic objects (robot bodies, manipulated objects) from Newton's
        physics simulation to OVRTX's render state using GPU kernels for efficiency.
        """
        if self._object_binding is None or self._object_newton_indices is None:
            return
        
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager
            
            # Get Newton physics state
            newton_state = NewtonManager.get_state_0()
            if newton_state is None:
                return
            
            # Map OVRTX transforms and update from Newton
            with self._object_binding.map(device=Device.CUDA, device_id=0) as attr_mapping:
                ovrtx_transforms = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)
                
                # Launch kernel to sync transforms
                wp.launch(
                    kernel=sync_newton_transforms_kernel,
                    dim=len(self._object_newton_indices),
                    inputs=[ovrtx_transforms, self._object_newton_indices, newton_state.body_q],
                    device=DEVICE,
                )
                # Unmap will commit the changes
                
        except Exception as e:
            # Silently fail to avoid spamming console
            if self._frame_counter == 1:
                print(f"[OVRTX] Warning: Failed to update object transforms: {e}")

    def step(self):
        """Step the renderer."""
        # The actual rendering happens in render()
        # This is called each simulation step but we don't need to do anything here
        pass

    def reset(self):
        """Reset the renderer."""
        if self._renderer:
            self._renderer.reset(time=0.0)

    def close(self):
        """Close the renderer and release resources."""
        if self._camera_binding:
            try:
                self._camera_binding.unbind()
            except Exception as e:
                print(f"Warning: Error unbinding camera transforms: {e}")
            self._camera_binding = None
        if self._object_binding:
            try:
                self._object_binding.unbind()
            except Exception as e:
                print(f"Warning: Error unbinding object transforms: {e}")
            self._object_binding = None

        if self._renderer:
            # Remove any USD content we added
            if self._usd_handles is not None:
                for handle in self._usd_handles:
                    try:
                        self._renderer.remove_usd(handle)
                    except Exception as e:
                        print(f"Warning: Error removing USD: {e}")
                self._usd_handles.clear()
            
            # Renderer cleanup is handled automatically by __del__
            self._renderer = None
        
        # Clear render product paths
        self._render_product_paths.clear()
        self._initialized_scene = False
