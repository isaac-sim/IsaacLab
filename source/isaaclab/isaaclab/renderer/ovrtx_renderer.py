# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX Renderer implementation."""

import math
import os
from dataclasses import MISSING
from pathlib import Path

import numpy as np
import torch
import warp as wp
from PIL import Image

# Set environment variables for OVRTX
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

from ovrtx import Renderer, RendererConfig

from isaaclab.utils.math import convert_camera_frame_orientation_convention

from .ovrtx_renderer_cfg import OVRTXRendererCfg
from .renderer import RendererBase


@wp.kernel
def _create_camera_transforms_kernel(
    positions: wp.array(dtype=wp.vec3),  # type: ignore
    orientations: wp.array(dtype=wp.quatf),  # type: ignore
    transforms: wp.array(dtype=wp.mat44d),  # type: ignore
):
    """Kernel to create camera transforms from positions and orientations.

    Args:
        positions: Array of camera positions, shape (num_cameras,)
        orientations: Array of camera orientations, shape (num_cameras,)
        transforms: Output array of camera transforms, shape (num_cameras,)
    """
    i = wp.tid()
    # Convert warp quaternion to rotation matrix and combine with translation
    pos = positions[i]
    quat = orientations[i]
    
    # Quaternion to rotation matrix (3x3)
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

    # Row 0
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qw * qz)
    r02 = 2.0 * (qx * qz + qw * qy)
    
    # Row 1
    r10 = 2.0 * (qx * qy + qw * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qw * qx)
    
    # Row 2
    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    
    # Build 4x4 homogeneous transform matrix
    # IMPORTANT: Matrix is stored in COLUMN-MAJOR order for OVRTX
    # So we transpose the rotation part: columns become rows
    _0 = wp.float64(0.0)
    _1 = wp.float64(1.0)
    # Note: Type issues with warp vec3 indexing are expected
    transforms[i] = wp.mat44d(  # type: ignore
        wp.float64(r00), wp.float64(r10), wp.float64(r20), _0,
        wp.float64(r01), wp.float64(r11), wp.float64(r21), _0,
        wp.float64(r02), wp.float64(r12), wp.float64(r22), _0,
        wp.float64(float(pos[0])), wp.float64(float(pos[1])), wp.float64(float(pos[2])), _1
    )


@wp.kernel
def _extract_tile_from_tiled_buffer_kernel(
    tiled_buffer: wp.array(dtype=wp.uint8, ndim=3),  # type: ignore (tiled_height, tiled_width, 4)
    tile_buffer: wp.array(dtype=wp.uint8, ndim=3),  # type: ignore (height, width, 4)
    tile_x: int,
    tile_y: int,
    tile_width: int,
    tile_height: int,
):
    """Extract a single tile from a tiled buffer.
    
    Args:
        tiled_buffer: Input tiled buffer, shape (tiled_height, tiled_width, 4)
        tile_buffer: Output buffer for single tile, shape (tile_height, tile_width, 4)
        tile_x: Tile position in x (horizontal)
        tile_y: Tile position in y (vertical)
        tile_width: Width of each tile
        tile_height: Height of each tile
    """
    y, x = wp.tid()
    
    # Calculate source position in tiled buffer
    src_x = tile_x * tile_width + x
    src_y = tile_y * tile_height + y
    
    # Copy RGBA channels
    tile_buffer[y, x, 0] = tiled_buffer[src_y, src_x, 0]
    tile_buffer[y, x, 1] = tiled_buffer[src_y, src_x, 1]
    tile_buffer[y, x, 2] = tiled_buffer[src_y, src_x, 2]
    tile_buffer[y, x, 3] = tiled_buffer[src_y, src_x, 3]


@wp.kernel
def _extract_depth_tile_from_tiled_buffer_kernel(
    tiled_buffer: wp.array(dtype=wp.float32, ndim=2),  # type: ignore (tiled_height, tiled_width)
    tile_buffer: wp.array(dtype=wp.float32, ndim=3),  # type: ignore (height, width, 1)
    tile_x: int,
    tile_y: int,
    tile_width: int,
    tile_height: int,
):
    """Extract a single depth tile from a tiled depth buffer.
    
    Args:
        tiled_buffer: Input tiled depth buffer, shape (tiled_height, tiled_width)
        tile_buffer: Output buffer for single tile, shape (tile_height, tile_width, 1)
        tile_x: Tile position in x (horizontal)
        tile_y: Tile position in y (vertical)
        tile_width: Width of each tile
        tile_height: Height of each tile
    """
    y, x = wp.tid()
    
    # Calculate source position in tiled buffer
    src_x = tile_x * tile_width + x
    src_y = tile_y * tile_height + y
    
    # Copy depth value
    tile_buffer[y, x, 0] = tiled_buffer[src_y, src_x]


@wp.kernel
def _sync_newton_transforms_kernel(
    ovrtx_transforms: wp.array(dtype=wp.mat44d),  # type: ignore
    newton_body_indices: wp.array(dtype=wp.int32),  # type: ignore
    newton_body_q: wp.array(dtype=wp.transformf),  # type: ignore
):
    """Kernel to sync Newton physics transforms to OVRTX render transforms.
    
    Converts Newton's wp.transformf (position + quaternion) to OVRTX's wp.mat44d
    (4x4 column-major matrix) for each object in the scene.
    
    Args:
        ovrtx_transforms: Output array of OVRTX transforms, shape (num_objects,)
        newton_body_indices: Newton body indices for each object, shape (num_objects,)
        newton_body_q: Newton body transforms from state.body_q, shape (num_bodies,)
    """
    i = wp.tid()
    body_idx = newton_body_indices[i]
    transform = newton_body_q[body_idx]
    
    # Use warp's built-in conversion and transpose for column-major format
    ovrtx_transforms[i] = wp.transpose(wp.mat44d(wp.math.transform_to_matrix(transform)))


_DEVICE = "cuda:0"


def _normalize_depth_to_uint8(depth_np: np.ndarray) -> tuple[np.ndarray, float | None, float | None]:
    """Normalize depth array to uint8 [0, 255] for visualization; invalid (inf/nan) -> 0.
    Returns (normalized_uint8, depth_min, depth_max); min/max are None if no valid pixels.
    """
    depth_valid = np.isfinite(depth_np)
    depth_min = depth_max = None
    if np.any(depth_valid):
        depth_min = float(depth_np[depth_valid].min())
        depth_max = float(depth_np[depth_valid].max())
    depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)
    if depth_min is not None and depth_max is not None and depth_max > depth_min:
        depth_normalized[depth_valid] = (
            (depth_np[depth_valid] - depth_min) / (depth_max - depth_min) * 255
        ).astype(np.uint8)
    return depth_normalized, depth_min, depth_max


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
        self._use_ovrtx_cloning = getattr(cfg, "use_ovrtx_cloning", True)
        self._image_folder = getattr(cfg, "image_folder", None)
        if self._image_folder:
            # Create the output directory if it doesn't exist
            Path(self._image_folder).mkdir(parents=True, exist_ok=True)
            print(f"[OVRTX] Images will be saved to: {self._image_folder}")

    def _deactivate_cloned_envs(self, stage) -> list:
        """Deactivate all cloned environments (env_1 onwards) to exclude from export.
        
        This optimization dramatically speeds up OvRTX initialization by exporting
        only the base environment (env_0), then using OvRTX's clone_usd() to replicate it.
        
        Args:
            stage: USD Stage containing environments
            
        Returns:
            List of deactivated prims (for reactivation later)
        """
        from pxr import Usd
        
        deactivated = []
        print(f"[OVRTX OPTIMIZE] Deactivating {self._num_envs - 1} cloned environments...")
        
        for env_idx in range(1, self._num_envs):  # Start from env_1
            env_path = f"/World/envs/env_{env_idx}"
            prim = stage.GetPrimAtPath(env_path)
            if prim.IsValid() and prim.IsActive():
                prim.SetActive(False)
                deactivated.append(prim)
                if env_idx <= 3 or env_idx == self._num_envs - 1:
                    print(f"  Deactivated: {env_path}")
        
        if self._num_envs > 5:
            print(f"  ... (deactivated {len(deactivated)} environments total)")
        
        return deactivated
    
    def _reactivate_prims(self, prims: list):
        """Reactivate previously deactivated prims.
        
        Args:
            prims: List of prims to reactivate
        """
        print(f"[OVRTX OPTIMIZE] Reactivating {len(prims)} environments...")
        for prim in prims:
            if prim.IsValid():
                prim.SetActive(True)
    
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
                semantic="token_string",
                prim_mode="must_exist",
            )
            
            if env_partition_binding is not None:
                # Create warp array with token strings
                # Note: OvRTX expects string tokens as a specific format
                partition_array = wp.array(partition_tokens, dtype=str, device="cpu")
                self._renderer.write_attribute(env_partition_binding, partition_array, sync=True)
                print(f"  ✓ Written primvars:omni:scenePartition to {self._num_envs} environments")
            else:
                print(f"  ⚠ Warning: Failed to bind primvars:omni:scenePartition on environments")
            
            # Write omni:scenePartition to camera prims
            cam_partition_binding = self._renderer.bind_attribute(
                prim_paths=camera_prim_paths,
                attribute_name="omni:scenePartition",
                semantic="token_string",
                prim_mode="must_exist",
            )
            
            if cam_partition_binding is not None:
                # Reuse the same partition tokens for cameras
                cam_partition_array = wp.array(partition_tokens, dtype=str, device="cpu")
                self._renderer.write_attribute(cam_partition_binding, cam_partition_array, sync=True)
                print(f"  ✓ Written omni:scenePartition to {self._num_envs} cameras")
            else:
                print(f"  ⚠ Warning: Failed to bind omni:scenePartition on cameras")
                
        except Exception as e:
            print(f"  ⚠ Warning: Failed to write scene partitions: {e}")
            import traceback
            traceback.print_exc()


    def initialize(self, usd_scene_path: str | None = None):
        """Initialize the OVRTX renderer with optional environment cloning.
        
        Two initialization modes based on use_ovrtx_cloning flag:
        
        Mode 1: OVRTX Internal Cloning (use_ovrtx_cloning=True, default):
        1. Load USD file containing only base environment (env_0)
        2. Use OvRTX clone_usd() to replicate environments (fast: O(1) or O(log N))
        3. ~10-100x faster initialization for 50+ environments
        
        Mode 2: Fully Cloned USD (use_ovrtx_cloning=False):
        1. Load USD file containing all N environments (fully cloned)
        2. No internal cloning needed
        3. Slower but may be useful for debugging or special rendering requirements
        
        The Isaac Sim stage is never modified - this only affects the exported USD.
        
        Args:
            usd_scene_path: Optional path to USD scene to load as root layer.
                           If provided, cameras will be injected into this file.
                           If not provided, cameras are created as the root layer.
        """
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
            # Note: tiled_camera.py may have filtered this to only contain env_0
            # if use_ovrtx_cloning=True and num_envs > 1
            print(f"[OVRTX] Injecting camera definitions...")
            combined_usd_path = self._inject_cameras_into_usd(usd_scene_path)
            
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
            
            # Clone base environment to all other environments in OvRTX (if enabled and needed)
            if self._use_ovrtx_cloning and self._num_envs > 1:
                print(f"[OVRTX] Using OVRTX internal cloning (use_ovrtx_cloning=True)")
                self._clone_environments_in_ovrtx()
                # Update scene partition attributes on cloned environments, objects, and cameras
                self._update_scene_partitions_after_clone(combined_usd_path)
            else:
                if self._num_envs > 1:
                    print(f"[OVRTX] Using fully cloned USD stage (use_ovrtx_cloning=False)")
                    print(f"   ✓ All {self._num_envs} environments loaded from USD")
            
            self._initialized_scene = True
            
            # Create binding for camera transforms (all environments now exist in OVRTX)
            camera_paths = [f"/World/envs/env_{i}/Camera" for i in range(self._num_envs)]
            self._camera_binding = self._renderer.bind_attribute(
                prim_paths=camera_paths,
                attribute_name="omni:xform",
                semantic="transform_4x4",
                prim_mode="must_exist",
            )
            
            if self._camera_binding is not None:
                print(f"  ✓ Camera binding created successfully")
            else:
                print(f"  ✗ WARNING: Camera binding is None!")
            
            # Setup object bindings for Newton physics sync
            self._setup_object_bindings()
        else:
            pass  # No USD scene: cameras as root layer not implemented
    
    def _inject_cameras_into_usd(self, usd_scene_path: str) -> str:
        """Inject camera and render product definitions into an existing USD file.
        
        Args:
            usd_scene_path: Path to the USD scene file
            
        Returns:
            Path to the combined USD file with cameras injected
        """
        import tempfile
        
        # Read the original USD
        with open(usd_scene_path, 'r') as f:
            original_usd = f.read()
        
        # Generate camera USD content (as a top-level Render scope)
        camera_parts = []
        camera_parts.append('\ndef Scope "Render"\n{\n')
        
        # Collect all camera paths
        camera_paths = [f"/World/envs/env_{env_idx}/Camera" for env_idx in range(self._num_envs)]
        
        # Create a SINGLE RenderProduct that references all cameras
        render_product_name = "RenderProduct"
        render_product_path = f"/Render/{render_product_name}"
        self._render_product_paths.append(render_product_path)
        
        # Build the camera relationship list: rel camera = [<path1>, <path2>, ...]
        camera_rel_list = ", ".join([f"<{path}>" for path in camera_paths])
        
        # Determine which RenderVar to use based on requested data types
        # Priority: depth > albedo/semantic > rgb (to optimize rendering performance)
        use_depth = any(dt in ["depth", "distance_to_image_plane", "distance_to_camera"] for dt in self._data_types)
        use_albedo = "albedo" in self._data_types
        use_semantic = "semantic_segmentation" in self._data_types
        use_rgb = any(dt in ["rgb", "rgba"] for dt in self._data_types)
        
        # Determine the primary render mode based on requested data types
        # Note: For now, we support single RenderVar. Multiple RenderVars would require orderedVars
        if use_depth and not (use_rgb or use_albedo or use_semantic):
            render_var_path = "/Render/Vars/depth"
            render_var_name = "depth"
            print(f"  Rendering mode: depth only")
        elif use_albedo and not (use_rgb or use_semantic):
            render_var_path = "/Render/Vars/albedo"
            render_var_name = "albedo"
            print(f"  Rendering mode: albedo only")
        elif use_semantic and not (use_rgb or use_albedo):
            render_var_path = "/Render/Vars/semantic"
            render_var_name = "semantic"
            print(f"  Rendering mode: semantic segmentation only")
        else:
            # Use SimpleShadingSD in simple shading mode, LdrColor otherwise
            if self._simple_shading_mode:
                render_var_path = "/Render/Vars/SimpleShading"
                render_var_name = "SimpleShading"
                print(f"  Rendering mode: RGB/RGBA (simple shading)")
            else:
                render_var_path = "/Render/Vars/LdrColor"
                render_var_name = "LdrColor"
                print(f"  Rendering mode: RGB/RGBA (full RTX)")
        
        camera_parts.append(f'''
    def RenderProduct "{render_product_name}" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [{camera_rel_list}]
        token omni:rtx:background:source:type = "domeLight"
        token omni:rtx:rendermode = "RealTimePathTracing"
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = <{render_var_path}>
        uniform int2 resolution = ({self._tiled_width}, {self._tiled_height})
    }}
''')
        
        # Add only the RenderVar that's actually being used
        # Determine sourceName based on render_var_name
        # These sourceNames correspond to the AOV (Arbitrary Output Variable) names in the renderer
        if render_var_name == "depth":
            source_name = "DistanceToImagePlaneSD"
        elif render_var_name == "SimpleShading":
            source_name = "SimpleShadingSD"
        elif render_var_name == "LdrColor":
            source_name = "LdrColor"
        elif render_var_name == "albedo":
            source_name = "DiffuseAlbedoSD"
        elif render_var_name == "semantic":
            source_name = "SemanticSegmentationSD"
        else:
            source_name = render_var_name  # Fallback
        
        camera_parts.append(f'''
    def "Vars"
    {{
        def RenderVar "{render_var_name}"
        {{
            uniform string sourceName = "{source_name}"
        }}
    }}
''')
        
        camera_parts.append('}\n')
        camera_content = ''.join(camera_parts)
        
        # Simply append the Render scope to the end of the file
        # This is safe since USD files are declarative
        combined_usd = original_usd.rstrip() + '\n\n' + camera_content
        
        # Save to temp file
        Path("/tmp/ovrtx_test").mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.usda', delete=False, dir='/tmp/ovrtx_test') as f:
            f.write(combined_usd)
            temp_path = f.name
        
        print(f"   Created combined USD: {temp_path}")
        return temp_path
    
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
                semantic="transform_4x4",
                prim_mode="must_exist",
            )
            
            if self._object_binding is not None:
                print(f"  ✓ Object binding created successfully")
                # Store Newton body indices for later lookup
                self._object_newton_indices = wp.array(newton_indices, dtype=wp.int32, device=_DEVICE)
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
                (self._num_envs, self._height, self._width, 4), dtype=wp.uint8, device=_DEVICE
            )
            # Create RGB view that references the same underlying array as RGBA, but only first 3 channels
            self._output_data_buffers["rgb"] = self._output_data_buffers["rgba"][:, :, :, :3]
        
        # Albedo buffer (4-channel RGBA format, similar to rgb/rgba)
        if "albedo" in self._data_types:
            self._output_data_buffers["albedo"] = wp.zeros(
                (self._num_envs, self._height, self._width, 4), dtype=wp.uint8, device=_DEVICE
            )
        
        # Semantic segmentation buffer (4-channel RGBA format for colorized output)
        if "semantic_segmentation" in self._data_types:
            self._output_data_buffers["semantic_segmentation"] = wp.zeros(
                (self._num_envs, self._height, self._width, 4), dtype=wp.uint8, device=_DEVICE
            )
        
        # Depth buffers (note: "depth" is an alias for "distance_to_image_plane")
        if "depth" in self._data_types:
            self._output_data_buffers["depth"] = wp.zeros(
                (self._num_envs, self._height, self._width, 1), dtype=wp.float32, device=_DEVICE
            )
        
        if "distance_to_image_plane" in self._data_types:
            self._output_data_buffers["distance_to_image_plane"] = wp.zeros(
                (self._num_envs, self._height, self._width, 1), dtype=wp.float32, device=_DEVICE
            )
        
        if "distance_to_camera" in self._data_types:
            self._output_data_buffers["distance_to_camera"] = wp.zeros(
                (self._num_envs, self._height, self._width, 1), dtype=wp.float32, device=_DEVICE
            )

    def _extract_rgba_tiles(
        self, tiled_data: wp.array, buffer_key: str, suffix: str = ""
    ) -> None:
        """Extract per-env RGBA tiles from tiled buffer and optionally save to disk."""
        for env_idx in range(self._num_envs):
            tile_x = env_idx % self._num_cols
            tile_y = env_idx // self._num_cols
            wp.launch(
                kernel=_extract_tile_from_tiled_buffer_kernel,
                dim=(self._height, self._width),
                inputs=[
                    tiled_data,
                    self._output_data_buffers[buffer_key][env_idx],
                    tile_x,
                    tile_y,
                    self._width,
                    self._height,
                ],
                device=_DEVICE,
            )
            self._save_image_to_disk(
                self._output_data_buffers[buffer_key][env_idx], env_idx, suffix=suffix
            )

    def _extract_depth_tiles(self, tiled_depth_data: wp.array) -> None:
        """Extract per-env depth tiles and populate all depth-type buffers; save depth images."""
        for env_idx in range(self._num_envs):
            tile_x = env_idx % self._num_cols
            tile_y = env_idx // self._num_cols
            for depth_type in ["depth", "distance_to_image_plane", "distance_to_camera"]:
                if depth_type in self._output_data_buffers:
                    wp.launch(
                        kernel=_extract_depth_tile_from_tiled_buffer_kernel,
                        dim=(self._height, self._width),
                        inputs=[
                            tiled_depth_data,
                            self._output_data_buffers[depth_type][env_idx],
                            tile_x,
                            tile_y,
                            self._width,
                            self._height,
                        ],
                        device=_DEVICE,
                    )
            if "depth" in self._output_data_buffers:
                self._save_depth_image_to_disk(
                    self._output_data_buffers["depth"][env_idx], env_idx
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
        camera_transforms = wp.zeros(num_envs, dtype=wp.mat44d, device=_DEVICE)
        
        # Launch kernel to populate transforms
        wp.launch(
            kernel=_create_camera_transforms_kernel,
            dim=num_envs,
            inputs=[camera_positions_wp, camera_orientations_wp, camera_transforms],
            device=_DEVICE,
        )
        
        # Update camera transforms in the scene using the binding
        if self._camera_binding is not None:
            with self._camera_binding.map(device="cuda", device_id=0) as attr_mapping:
                wp_transforms_view = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)
                wp.copy(wp_transforms_view, camera_transforms)
                # Unmap will commit the changes
        
        # Update object transforms from Newton physics
        self._update_object_transforms()
        
        # Step the renderer to produce frames (single RenderProduct, tiled output)
        if self._renderer is not None and len(self._render_product_paths) > 0:
            try:
                # Render using the single render product
                render_product_set = set(self._render_product_paths)
                
                products = self._renderer.step(
                    render_products=render_product_set,
                    delta_time=1.0 / 60.0,
                )

                # Extract rendered images from the single render product
                # The product should contain a single tiled frame
                product_path = self._render_product_paths[0]
                if product_path in products:
                    product = products[product_path]
                    
                    if len(product.frames) > 0:
                        frame = product.frames[0]

                        # Extract RGB/RGBA
                        rgb_render_var = None
                        if "SimpleShadingSD" in frame.render_vars:
                            rgb_render_var = "SimpleShadingSD"
                        elif "LdrColor" in frame.render_vars:
                            rgb_render_var = "LdrColor"
                        
                        if rgb_render_var and "rgba" in self._output_data_buffers:
                            with frame.render_vars[rgb_render_var].map(device="cuda") as mapping:
                                tiled_data = wp.from_dlpack(mapping.tensor)
                                self._save_tiled_image_to_disk(tiled_data, suffix="rgb")
                                self._extract_rgba_tiles(tiled_data, "rgba", suffix="rgb")
                        
                        # Extract depth if available
                        # Check for depth render vars by their sourceName (DistanceToImagePlaneSD or DepthSD)
                        depth_source_names = ["DistanceToImagePlaneSD", "DepthSD"]
                        depth_var_found = None
                        for source_name in depth_source_names:
                            if source_name in frame.render_vars:
                                depth_var_found = source_name
                                break
                        
                        if depth_var_found:
                            with frame.render_vars[depth_var_found].map(device="cuda") as mapping:
                                tiled_depth_data = wp.from_dlpack(mapping.tensor)

                                if tiled_depth_data.dtype == wp.uint32:
                                    depth_torch = wp.to_torch(tiled_depth_data)
                                    tiled_depth_data = wp.from_torch(
                                        depth_torch.view(torch.float32), dtype=wp.float32
                                    )

                                self._save_tiled_depth_image_to_disk(tiled_depth_data)
                                self._extract_depth_tiles(tiled_depth_data)
                        
                        if "DiffuseAlbedoSD" in frame.render_vars and "albedo" in self._output_data_buffers:
                            with frame.render_vars["DiffuseAlbedoSD"].map(device="cuda") as mapping:
                                tiled_albedo_data = wp.from_dlpack(mapping.tensor)
                                self._save_tiled_image_to_disk(tiled_albedo_data, suffix="albedo")
                                self._extract_rgba_tiles(tiled_albedo_data, "albedo", suffix="albedo")
                        
                        # Extract semantic segmentation if available
                        if "SemanticSegmentationSD" in frame.render_vars and "semantic_segmentation" in self._output_data_buffers:
                            with frame.render_vars["SemanticSegmentationSD"].map(device="cuda") as mapping:
                                tiled_semantic_data = wp.from_dlpack(mapping.tensor)

                                if tiled_semantic_data.dtype == wp.uint32:
                                    semantic_torch = wp.to_torch(tiled_semantic_data)
                                    semantic_uint8_torch = semantic_torch.view(torch.uint8)
                                    if len(semantic_torch.shape) == 2:
                                        h, w = semantic_torch.shape
                                        semantic_uint8_torch = semantic_uint8_torch.reshape(h, w, 4)
                                    tiled_semantic_data = wp.from_torch(semantic_uint8_torch, dtype=wp.uint8)

                                self._save_tiled_image_to_disk(tiled_semantic_data, suffix="semantic")
                                self._extract_rgba_tiles(
                                    tiled_semantic_data, "semantic_segmentation", suffix="semantic"
                                )

            except Exception as e:
                print(f"Warning: OVRTX rendering failed: {e}")
                import traceback
                traceback.print_exc()
                # Keep the output buffers as-is (zeros from initialization)

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
            with self._object_binding.map(device="cuda", device_id=0) as attr_mapping:
                ovrtx_transforms = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)
                
                # Launch kernel to sync transforms
                wp.launch(
                    kernel=_sync_newton_transforms_kernel,
                    dim=len(self._object_newton_indices),
                    inputs=[ovrtx_transforms, self._object_newton_indices, newton_state.body_q],
                    device=_DEVICE,
                )
                # Unmap will commit the changes
                
        except Exception as e:
            # Silently fail to avoid spamming console
            if self._frame_counter == 1:
                print(f"[OVRTX] Warning: Failed to update object transforms: {e}")

    def _save_image_to_disk(self, rendered_data_wp: wp.array, env_idx: int, suffix: str = ""):
        """Save rendered image to disk.
        
        Args:
            rendered_data_wp: Warp array containing RGBA data, shape (height, width, 4)
            env_idx: Environment index for filename
            suffix: Optional suffix to add to filename (e.g., "albedo", "semantic", "rgb")
        """
        # Only save if image_folder is configured
        if not self._image_folder:
            return
            
        try:
            # Convert warp array to torch tensor, then to numpy
            rendered_data_torch = wp.to_torch(rendered_data_wp)
            rendered_data_np = rendered_data_torch.cpu().numpy()
            
            # Convert from float [0, 1] to uint8 [0, 255]
            if rendered_data_np.dtype in [np.float32, np.float64]:
                rendered_data_np = (rendered_data_np * 255).astype(np.uint8)
            
            # Use the configured image folder
            output_dir = Path(self._image_folder)
            
            # Save as PNG
            # rendered_data_np is shape (height, width, 4) for RGBA
            if len(rendered_data_np.shape) == 3 and rendered_data_np.shape[2] == 4:
                # RGBA image
                image = Image.fromarray(rendered_data_np, mode='RGBA')
            elif len(rendered_data_np.shape) == 3 and rendered_data_np.shape[2] == 3:
                # RGB image
                image = Image.fromarray(rendered_data_np, mode='RGB')
            elif len(rendered_data_np.shape) == 2:
                # Grayscale image
                image = Image.fromarray(rendered_data_np, mode='L')
            else:
                print(f"Warning: Unexpected image shape {rendered_data_np.shape}, cannot save")
                return
            
            # Save with frame, environment index, and suffix in filename
            if suffix:
                output_path = output_dir / f"{suffix}_frame_{self._frame_counter:06d}_env_{env_idx:04d}.png"
            else:
                output_path = output_dir / f"frame_{self._frame_counter:06d}_env_{env_idx:04d}.png"
            image.save(output_path)
            
            # Only print for first environment and first few frames to avoid spam
            if env_idx == 0 and self._frame_counter <= 5:
                print(f"[OVRTX] Saved rendered image: {output_path}")
                
        except Exception as e:
            print(f"Warning: Failed to save image for env {env_idx}: {e}")
    
    def _save_depth_image_to_disk(self, depth_data_wp: wp.array, env_idx: int):
        """Save depth image to disk as grayscale PNG (normalized).
        
        Args:
            depth_data_wp: Warp array containing depth data, shape (height, width, 1)
            env_idx: Environment index for filename
        """
        # Only save if image_folder is configured
        if not self._image_folder:
            return
            
        try:
            # Convert warp array to torch tensor, then to numpy
            depth_data_torch = wp.to_torch(depth_data_wp)
            depth_data_np = depth_data_torch.cpu().numpy()
            
            if len(depth_data_np.shape) == 3 and depth_data_np.shape[2] == 1:
                depth_data_np = depth_data_np[:, :, 0]

            depth_normalized, depth_min, depth_max = _normalize_depth_to_uint8(depth_data_np)
            output_dir = Path(self._image_folder)
            image = Image.fromarray(depth_normalized, mode="L")
            output_path = output_dir / f"depth_frame_{self._frame_counter:06d}_env_{env_idx:04d}.png"
            image.save(output_path)
            if env_idx == 0 and self._frame_counter <= 5 and depth_min is not None and depth_max is not None:
                print(f"[OVRTX] Saved depth image: {output_path} (range: {depth_min:.3f} to {depth_max:.3f})")
                
        except Exception as e:
            print(f"Warning: Failed to save depth image for env {env_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_tiled_image_to_disk(self, tiled_data_wp: wp.array, suffix: str = ""):
        """Save tiled image (all environments in a grid) to disk.
        
        Args:
            tiled_data_wp: Warp array containing tiled RGBA data, shape (tiled_height, tiled_width, 4)
            suffix: Optional suffix to add to filename (e.g., "albedo", "semantic", "rgb")
        """
        # Only save if image_folder is configured
        if not self._image_folder:
            return
            
        try:
            # Convert warp array to torch tensor, then to numpy
            tiled_data_torch = wp.to_torch(tiled_data_wp)
            tiled_data_np = tiled_data_torch.cpu().numpy()
            
            # Convert from float [0, 1] to uint8 [0, 255]
            if tiled_data_np.dtype in [np.float32, np.float64]:
                tiled_data_np = (tiled_data_np * 255).astype(np.uint8)
            
            # Use the configured image folder
            output_dir = Path(self._image_folder)
            
            # Save as PNG
            image = Image.fromarray(tiled_data_np, mode='RGBA')
            if suffix:
                output_path = output_dir / f"{suffix}_frame_{self._frame_counter:06d}_tiled.png"
            else:
                output_path = output_dir / f"frame_{self._frame_counter:06d}_tiled.png"
            image.save(output_path)
            
            # Print only for first few frames
            if self._frame_counter <= 5:
                print(f"[OVRTX] Saved tiled {suffix + ' ' if suffix else ''}image ({self._num_envs} envs in {self._num_cols}x{self._num_rows} grid): {output_path}")
                
        except Exception as e:
            print(f"Warning: Failed to save tiled image: {e}")
            import traceback
            traceback.print_exc()

    def _save_tiled_depth_image_to_disk(self, tiled_depth_data_wp: wp.array):
        """Save tiled depth image (all environments in a grid) to disk.
        
        Args:
            tiled_depth_data_wp: Warp array containing tiled depth data, shape (tiled_height, tiled_width)
        """
        # Only save if image_folder is configured
        if not self._image_folder:
            return
            
        try:
            # Convert warp array to torch tensor, then to numpy
            tiled_depth_torch = wp.to_torch(tiled_depth_data_wp)
            tiled_depth_np = tiled_depth_torch.cpu().numpy()
            depth_normalized, depth_min, depth_max = _normalize_depth_to_uint8(tiled_depth_np)
            output_dir = Path(self._image_folder)
            image = Image.fromarray(depth_normalized, mode="L")
            output_path = output_dir / f"depth_frame_{self._frame_counter:06d}_tiled.png"
            image.save(output_path)
            if self._frame_counter <= 5 and depth_min is not None and depth_max is not None:
                print(f"[OVRTX] Saved tiled depth image ({self._num_envs} envs in {self._num_cols}x{self._num_rows} grid): {output_path} (range: {depth_min:.3f} to {depth_max:.3f})")
                
        except Exception as e:
            print(f"Warning: Failed to save tiled depth image: {e}")
            import traceback
            traceback.print_exc()

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
