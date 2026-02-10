# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

import math
import torch

import warp as wp
from newton.sensors import SensorTiledCamera

from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from .newton_warp_renderer_cfg import NewtonWarpRendererCfg
from .renderer import RendererBase


@wp.kernel
def _create_camera_transforms_kernel(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quatf),
    transforms: wp.array(dtype=wp.transformf, ndim=2),
):
    """Kernel to create camera transforms from positions and orientations.

    Args:
        positions: Array of camera positions, shape (num_cameras,)
        orientations: Array of camera orientations, shape (num_cameras,)
        transforms: Output array of camera transforms, shape (num_cameras, 1)
    """
    i = wp.tid()
    transforms[i, 0] = wp.transformf(positions[i], orientations[i])


@wp.kernel
def _convert_raw_rgb_tiled(
    raw_buffer: wp.array(dtype=wp.uint32, ndim=3),
    output_buffer: wp.array(dtype=wp.uint8, ndim=3),
    image_width: int,
    image_height: int,
    num_tiles_x: int,
):
    """Convert raw tiled RGB buffer (uint32 packed) to tiled RGBA (uint8). For debugging purposes.

    The raw buffer has shape (num_worlds num_cameras, width * height) where each uint32 encodes RGBA as 4 bytes.
    This kernel converts it to the tiled format (tiled_height, tiled_width, 4) of uint8.

    Args:
        raw_buffer: Input raw buffer from SensorTiledCamera, shape (num_worlds, num_cameras, width * height) of uint32
        output_buffer: Output buffer in tiled format (tiled_height, tiled_width, 4) of uint8
        image_width: Width of each camera image
        image_height: Height of each camera image
        num_tiles_x: Number of tiles in x-direction (horizontal)
    """
    y, x = wp.tid()

    # Determine which tile and which pixel within the tile
    # x is width (horizontal), y is height (vertical)
    tile_x = x // image_width
    tile_y = y // image_height
    pixel_x = x % image_width
    pixel_y = y % image_height

    # Compute camera ID from tile position
    camera_id = tile_y * num_tiles_x + tile_x

    # Compute the pixel index within this camera's buffer
    # The buffer is flattened as (width * height), so row-major indexing
    pixel_idx = pixel_y * image_width + pixel_x

    # Read the packed uint32 value from raw_buffer[camera_id, 0, pixel_idx]
    packed_color = raw_buffer[camera_id, 0, pixel_idx]

    # Compute output x coordinate
    output_x = tile_x * image_width + pixel_x

    # Unpack the uint32 into 4 uint8 values (RGBA channels)
    # Assuming little-endian byte order: ABGR format in memory
    output_buffer[y, output_x, 0] = wp.uint8((packed_color >> wp.uint32(0)) & wp.uint32(0xFF))  # R
    output_buffer[y, output_x, 1] = wp.uint8((packed_color >> wp.uint32(8)) & wp.uint32(0xFF))  # G
    output_buffer[y, output_x, 2] = wp.uint8((packed_color >> wp.uint32(16)) & wp.uint32(0xFF))  # B
    output_buffer[y, output_x, 3] = wp.uint8((packed_color >> wp.uint32(24)) & wp.uint32(0xFF))  # A


@wp.kernel
def _convert_raw_depth_tiled(
    raw_buffer: wp.array(dtype=wp.float32, ndim=3),
    output_buffer: wp.array(dtype=wp.float32, ndim=3),
    image_width: int,
    image_height: int,
    num_tiles_x: int,
):
    """Convert raw tiled depth buffer to tiled depth format. For debugging purposes.

    The raw buffer has shape (num_worlds, num_cameras, width * height) of float32 depth values.
    This kernel converts it to the tiled format (tiled_height, tiled_width, 1) of float32.

    Args:
        raw_buffer: Input raw buffer from SensorTiledCamera, shape (num_worlds, num_cameras, width * height) of float32
        output_buffer: Output buffer in tiled format (tiled_height, tiled_width, 1) of float32
        image_width: Width of each camera image
        image_height: Height of each camera image
        num_tiles_x: Number of tiles in x-direction (horizontal)
    """
    y, x = wp.tid()

    # Determine which tile and which pixel within the tile
    # x is width (horizontal), y is height (vertical)
    tile_x = x // image_width
    tile_y = y // image_height
    pixel_x = x % image_width
    pixel_y = y % image_height

    # Compute camera ID from tile position
    camera_id = tile_y * num_tiles_x + tile_x

    # Compute the pixel index within this camera's buffer
    # The buffer is flattened as (width * height), so row-major indexing
    pixel_idx = pixel_y * image_width + pixel_x

    # Compute output x coordinate
    output_x = tile_x * image_width + pixel_x

    # Copy the depth value from raw_buffer[camera_id, 0, pixel_idx]
    output_buffer[y, output_x, 0] = raw_buffer[camera_id, 0, pixel_idx]


@wp.kernel
def _detile_rgba_kernel(
    tiled_image: wp.array(dtype=wp.uint8, ndim=3),  # shape: (tiled_H, tiled_W, 4)
    output: wp.array(dtype=wp.uint8, ndim=4),  # shape: (num_envs, H, W, 4)
    tiles_per_side: int,
    tile_height: int,
    tile_width: int,
):
    """Detile a tiled RGBA image into separate environment images."""
    env_id, y, x = wp.tid()
    
    # Calculate which tile this environment corresponds to
    tile_y = env_id // tiles_per_side
    tile_x = env_id % tiles_per_side
    
    # Calculate position in tiled image
    tiled_y = tile_y * tile_height + y
    tiled_x = tile_x * tile_width + x
    
    # Copy RGBA channels
    output[env_id, y, x, 0] = tiled_image[tiled_y, tiled_x, 0]  # R
    output[env_id, y, x, 1] = tiled_image[tiled_y, tiled_x, 1]  # G
    output[env_id, y, x, 2] = tiled_image[tiled_y, tiled_x, 2]  # B
    output[env_id, y, x, 3] = tiled_image[tiled_y, tiled_x, 3]  # A


@wp.kernel
def _detile_depth_kernel(
    tiled_depth: wp.array(dtype=wp.float32, ndim=2),  # shape: (tiled_H, tiled_W)
    output: wp.array(dtype=wp.float32, ndim=4),  # shape: (num_envs, H, W, 1)
    tiles_per_side: int,
    tile_height: int,
    tile_width: int,
):
    """Detile a tiled depth image into separate environment depth images."""
    env_id, y, x = wp.tid()
    
    # Calculate which tile this environment corresponds to
    tile_y = env_id // tiles_per_side
    tile_x = env_id % tiles_per_side
    
    # Calculate position in tiled image
    tiled_y = tile_y * tile_height + y
    tiled_x = tile_x * tile_width + x
    
    # Copy depth value
    output[env_id, y, x, 0] = tiled_depth[tiled_y, tiled_x]


class NewtonWarpRenderer(RendererBase):
    """Newton Warp Renderer implementation."""

    _model = None

    # tiled camerae sensor from warp trace
    _tiled_camera_sensor = None

    def __init__(self, cfg: NewtonWarpRendererCfg):
        super().__init__(cfg)

    def initialize(self):
        """Initialize the renderer."""
        self._model = NewtonManager.get_model()

        self._tiled_camera_sensor = SensorTiledCamera(
            model=self._model,
            num_cameras=1,  # TODO: currently only supports 1 camera per world
            width=self._width,
            height=self._height,
            options=SensorTiledCamera.Options(colors_per_shape=True),
        )

        # Note: camera rays will be computed when we have access to TiledCamera
        # for now use default 45 degree FOV
        self._camera_rays = None

        # Initialize output buffers
        self._initialize_output()

    def set_camera_rays_from_intrinsics(self, intrinsic_matrices: torch.Tensor):
        """Set camera FOV from intrinsic matrices (vectorized for all cameras).

        Args:
            intrinsic_matrices: Camera intrinsic matrices of shape (num_cameras, 3, 3)
                             Format: [[f_x,   0, c_x],
                                     [  0, f_y, c_y],
                                     [  0,   0,   1]]
        """
        # Extract vertical focal lengths for all cameras (vectorized)
        # Shape: (num_cameras,)
        f_y_all = intrinsic_matrices[:, 1, 1]  # All cameras' vertical focal lengths in pixels

        # Calculate vertical FOV for all cameras (vectorized)
        # fov = 2 * atan(height / (2 * f_y))
        # Shape: (num_cameras,)
        fov_radians_all = 2.0 * torch.atan(self._height / (2.0 * f_y_all))

        # Convert to warp array
        fov_radians_wp = wp.from_torch(fov_radians_all, dtype=wp.float32)

        # Compute camera rays with per-camera FOVs (vectorized)
        # SensorTiledCamera.compute_pinhole_camera_rays accepts array of FOVs
        self._camera_rays = self._tiled_camera_sensor.compute_pinhole_camera_rays(fov_radians_wp)

    def _initialize_output(self):
        """Initialize the output of the renderer."""
        self._data_types = ["rgba", "rgb", "depth"]
        self._num_tiles_per_side = math.ceil(math.sqrt(self._num_envs))

        # Raw buffer to hold data from the tiled camera sensor
        self._raw_output_rgb_buffer = self._tiled_camera_sensor.create_color_image_output()
        self._raw_output_depth_buffer = self._tiled_camera_sensor.create_depth_image_output()

        self._output_data_buffers["rgba"] = wp.zeros(
            (self._num_envs, self._height, self._width, 4), dtype=wp.uint8, device=self._raw_output_rgb_buffer.device
        )
        # Create RGB view that references the same underlying array as RGBA, but only first 3 channels
        self._output_data_buffers["rgb"] = self._output_data_buffers["rgba"][:, :, :, :3]
        self._output_data_buffers["depth"] = wp.zeros(
            (self._num_envs, self._height, self._width, 1),
            dtype=wp.float32,
            device=self._raw_output_depth_buffer.device,
        )

    def render(
        self, camera_positions: torch.Tensor, camera_orientations: torch.Tensor, intrinsic_matrices: torch.Tensor
    ):
        """Render the scene.

        Args:
            camera_positions: Tensor of shape (num_envs, 3) - camera positions in world frame
            camera_orientations: Tensor of shape (num_envs, 4) - camera quaternions (x, y, z, w) in world frame
            intrinsic_matrices: Tensor of shape (num_envs, 3, 3) - camera intrinsic matrices
        """
        if self._camera_rays is None:
            self.set_camera_rays_from_intrinsics(intrinsic_matrices)
        num_envs = camera_positions.shape[0]

        # Convert torch tensors to warp arrays directly on GPU
        # Positions: shape (num_envs, 3) -> shape (num_envs,) of vec3
        camera_positions_wp = wp.from_torch(camera_positions.contiguous(), dtype=wp.vec3)
        camera_quats_converted = convert_camera_frame_orientation_convention(
            camera_orientations, origin="world", target="opengl"
        )

        camera_orientations_wp = wp.from_torch(camera_quats_converted, dtype=wp.quat)

        # Create camera transforms array, TODO: num_cameras = 1
        # Format: wp.array of shape (num_envs, num_cameras), dtype=wp.transformf
        camera_transforms = wp.empty((num_envs, 1), dtype=wp.transformf, device=camera_positions_wp.device)

        # Launch kernel to populate transforms (vectorized operation)
        wp.launch(
            kernel=_create_camera_transforms_kernel,
            dim=num_envs,
            inputs=[camera_positions_wp, camera_orientations_wp, camera_transforms],
            device=camera_positions_wp.device,
        )

        # Render using SensorTiledCamera
        self._tiled_camera_sensor.render(
            state=NewtonManager.get_state_0(),  # Use current physics state
            camera_transforms=camera_transforms,
            camera_rays=self._camera_rays,
            color_image=self._raw_output_rgb_buffer,
            depth_image=self._raw_output_depth_buffer,
        )

        # The tiled camera sensor outputs a tiled image grid
        # For num_envs cameras, it creates a sqrt(num_envs) x sqrt(num_envs) grid
        # Each tile is (height x width)
        tiles_per_side = math.ceil(math.sqrt(num_envs))
        tiled_height = tiles_per_side * self._height
        tiled_width = tiles_per_side * self._width
        
        # Raw RGB buffer is packed uint32 (RGBA), shape is (tiled_height, tiled_width)
        # Convert to RGBA uint8 view: shape becomes (tiled_height, tiled_width, 4)
        rgba_tiled = wp.array(
            ptr=self._raw_output_rgb_buffer.ptr,
            shape=(tiled_height, tiled_width, 4),
            dtype=wp.uint8,
            device=self._raw_output_rgb_buffer.device,
        )
        
        # Use detiling kernel to extract individual environment images
        actual_num_envs = min(num_envs, self._num_envs)
        wp.launch(
            kernel=_detile_rgba_kernel,
            dim=(actual_num_envs, self._height, self._width),
            inputs=[rgba_tiled, self._output_data_buffers["rgba"], tiles_per_side, self._height, self._width],
            device=rgba_tiled.device,
        )
        
        self._output_data_buffers["rgb"] = self._output_data_buffers["rgba"][:, :, :, :3]
        
        # Similar detiling for depth
        depth_tiled = self._raw_output_depth_buffer.reshape((tiled_height, tiled_width))
        
        wp.launch(
            kernel=_detile_depth_kernel,
            dim=(actual_num_envs, self._height, self._width),
            inputs=[depth_tiled, self._output_data_buffers["depth"], tiles_per_side, self._height, self._width],
            device=depth_tiled.device,
        )

    def step(self):
        """Step the renderer."""
        pass

    def reset(self):
        """Reset the renderer."""
        pass

    def close(self):
        """Close the renderer."""
        pass
