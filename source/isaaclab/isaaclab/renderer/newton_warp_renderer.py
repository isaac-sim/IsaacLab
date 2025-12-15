# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

import math
import torch

import warp as wp
from newton.sensors import TiledCameraSensor

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
    # Create transform and store in output array (camera i, world 0)
    # Apply rotation to vertically flip the image
    # This should fix the upside-down orientation
    # (0, 0, 0, 1)
    # transforms[i, 0] = wp.transformf(positions[i] + wp.vec3(10.0, 0.0, 0.0), wp.quat(-0.5, -0.5, -0.5, -0.5))
    transforms[i, 0] = wp.transformf(positions[i], orientations[i])
    # transforms[i, 0] = wp.transformf(positions[i], wp.quat(-0.5, 0.5, -0.5, 0.5))


@wp.kernel
def _unpack_uint32_to_rgba(
    input_buffer: wp.array(dtype=wp.uint32, ndim=3),
    output_buffer: wp.array(dtype=wp.uint8, ndim=4),
    image_width: int,
):
    """Unpack uint32 color values to RGBA uint8 channels.

    Converts from (num_cameras, 1, height*width) uint32 to (num_cameras, height, width, 4) uint8.
    Each uint32 value is unpacked into 4 uint8 channels (RGBA).

    Args:
        input_buffer: Input buffer of shape (num_cameras, 1, height*width) of uint32
        output_buffer: Output buffer of shape (num_cameras, height, width, 4) of uint8
        image_width: Width of the image (needed to compute pixel index)
    """
    camera_id, height_id, width_id = wp.tid()

    # Compute the pixel index in the flattened buffer (row-major: y * width + x)
    pixel_idx = height_id * image_width + width_id

    # Read the packed uint32 value from input_buffer[camera_id, 0, pixel_idx]
    packed_color = input_buffer[camera_id, 0, pixel_idx]

    # Unpack the uint32 into 4 uint8 values (RGBA channels)
    # Assuming little-endian byte order: ABGR format in memory
    output_buffer[camera_id, height_id, width_id, 0] = wp.uint8((packed_color >> wp.uint32(0)) & wp.uint32(0xFF))  # R
    output_buffer[camera_id, height_id, width_id, 1] = wp.uint8((packed_color >> wp.uint32(8)) & wp.uint32(0xFF))  # G
    output_buffer[camera_id, height_id, width_id, 2] = wp.uint8((packed_color >> wp.uint32(16)) & wp.uint32(0xFF))  # B
    output_buffer[camera_id, height_id, width_id, 3] = wp.uint8((packed_color >> wp.uint32(24)) & wp.uint32(0xFF))  # A
    # transforms[i, 0] = wp.transformf(positions[i] + wp.vec3(10.0, 0.0, 0.0), wp.quat(-0.5, 0.5, -0.5, 0.5))


@wp.kernel
def _convert_raw_rgb_tiled(
    raw_buffer: wp.array(dtype=wp.uint32, ndim=3),
    output_buffer: wp.array(dtype=wp.uint8, ndim=3),
    image_width: int,
    image_height: int,
    num_tiles_x: int,
):
    """Convert raw tiled RGB buffer (uint32 packed) to tiled RGBA (uint8).

    The raw buffer has shape (num_cameras, 1, width * height) where each uint32 encodes RGBA as 4 bytes.
    This kernel converts it to the tiled format (tiled_height, tiled_width, 4) of uint8.

    Args:
        raw_buffer: Input raw buffer from TiledCameraSensor, shape (num_cameras, 1, width * height) of uint32
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

    # Flip horizontally within each tile only (not the whole image)
    flipped_pixel_x = image_width - 1 - pixel_x
    output_x = tile_x * image_width + flipped_pixel_x

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
    """Convert raw tiled depth buffer to tiled depth format.

    The raw buffer has shape (num_cameras, 1, width * height) of float32 depth values.
    This kernel converts it to the tiled format (tiled_height, tiled_width, 1) of float32.

    Args:
        raw_buffer: Input raw buffer from TiledCameraSensor, shape (num_cameras, 1, width * height) of float32
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

    # Flip horizontally within each tile only (not the whole image)
    flipped_pixel_x = image_width - 1 - pixel_x
    output_x = tile_x * image_width + flipped_pixel_x

    # Copy the depth value from raw_buffer[camera_id, 0, pixel_idx]
    output_buffer[y, output_x, 0] = raw_buffer[camera_id, 0, pixel_idx]


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

        self._tiled_camera_sensor = TiledCameraSensor(
            model=self._model,
            num_cameras=1,  # 1 because model already has
            width=self._width,
            height=self._height,
            options=TiledCameraSensor.Options(colors_per_shape=True),
            # options=TiledCameraSensor.Options(
            #     default_light=True, default_light_shadows=True, colors_per_shape=True, checkerboard_texture=True
            # )
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
        # TiledCameraSensor.compute_pinhole_camera_rays accepts array of FOVs
        self._camera_rays = self._tiled_camera_sensor.compute_pinhole_camera_rays(fov_radians_wp)

    def _initialize_output(self):
        """Initialize the output of the renderer."""
        self._data_types = ["rgba", "depth"]
        self._num_tiles_per_side = math.ceil(math.sqrt(self._num_envs))

        # Output data buffers that are exposed to the user
        # After transpose, dimensions are swapped
        # Format: (tiled_height, tiled_width, channels) where height and width are swapped
        tiled_width = self._num_tiles_per_side * self._width
        tiled_height = self._num_tiles_per_side * self._height

        # Raw buffer to hold data from the tiled camera sensor
        self._raw_output_rgb_buffer = self._tiled_camera_sensor.create_color_image_output()
        self._raw_output_depth_buffer = self._tiled_camera_sensor.create_depth_image_output()

        # Output buffers with swapped dimensions due to transpose
        # self._output_data_buffers["rgba"] = wp.zeros((tiled_height, tiled_width, 4), dtype=wp.uint8, device=self._raw_output_rgb_buffer.device)
        # self._output_data_buffers["depth"] = wp.zeros((tiled_height, tiled_width, 1), dtype=wp.float32, device=self._raw_output_depth_buffer.device)

        self._output_data_buffers["rgba"] = wp.zeros(
            (self._num_envs, self._height, self._width, 4), dtype=wp.uint8, device=self._raw_output_rgb_buffer.device
        )
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
            camera_positions: Tensor of shape (num_cameras, 3) - camera positions in world frame
            camera_orientations: Tensor of shape (num_cameras, 4) - camera quaternions (w, x, y, z) in world frame
            intrinsic_matrices: Tensor of shape (num_cameras, 3, 3) - camera intrinsic matrices
        """
        if self._camera_rays is None:
            self.set_camera_rays_from_intrinsics(intrinsic_matrices)
        num_cameras = camera_positions.shape[0]

        # Convert torch tensors to warp arrays directly on GPU
        # Positions: shape (num_cameras, 3) -> shape (num_cameras,) of vec3
        camera_positions_wp = wp.from_torch(camera_positions.contiguous(), dtype=wp.vec3)

        camera_quats_converted = convert_camera_frame_orientation_convention(
            camera_orientations, origin="world", target="opengl"
        )

        # breakpoint()

        # Quaternions: need to reorder from (w,x,y,z) to (x,y,z,w) for warp
        # Create a copy with reordered quaternion components
        camera_quats_reordered = torch.stack(
            [
                camera_quats_converted[:, 1],  # x
                camera_quats_converted[:, 2],  # y
                camera_quats_converted[:, 3],  # z
                camera_quats_converted[:, 0],  # w
            ],
            dim=1,
        ).contiguous()  # Shape: (num_cameras, 4)

        # breakpoint()

        # (220, -210, 0)
        # (5, 0, 2) + (10, 0, 0)

        # breakpoint()

        camera_orientations_wp = wp.from_torch(camera_quats_reordered, dtype=wp.quat)

        # Create camera transforms array
        # Format: wp.array of shape (num_cameras, num_worlds=1), dtype=wp.transformf
        camera_transforms = wp.empty((num_cameras, 1), dtype=wp.transformf, device=camera_positions_wp.device)

        # Launch kernel to populate transforms (vectorized operation)
        wp.launch(
            kernel=_create_camera_transforms_kernel,
            dim=num_cameras,
            inputs=[camera_positions_wp, camera_orientations_wp, camera_transforms],
            device=camera_positions_wp.device,
        )

        # Render using TiledCameraSensor
        self._tiled_camera_sensor.render(
            state=NewtonManager.get_state_0(),  # Use current physics state
            camera_transforms=camera_transforms,
            camera_rays=self._camera_rays,
            color_image=self._raw_output_rgb_buffer,
            depth_image=self._raw_output_depth_buffer,
        )

        # breakpoint()

        # Convert uint32 to uint8 RGBA using kernel
        # Input: (num_cameras, 1, width*height) uint32
        # Output: (num_cameras, height, width, 4) uint8
        wp.launch(
            kernel=_unpack_uint32_to_rgba,
            dim=(self._num_envs, self._height, self._width),
            inputs=[
                self._raw_output_rgb_buffer,
                self._output_data_buffers["rgba"],
                self._width,
            ],
            device=self._raw_output_rgb_buffer.device,
        )

        # Reshape depth buffer: (num_cameras, 1, width*height) -> (num_cameras, height, width, 1)
        self._output_data_buffers["depth"] = self._raw_output_depth_buffer.reshape(
            (self._num_envs, self._height, self._width, 1)
        )

        # # Convert raw buffers to user-facing tiled format
        # tiled_width = self._num_tiles_per_side * self._width
        # tiled_height = self._num_tiles_per_side * self._height

        # # RGB: uint32 packed -> uint8 RGBA (tiled_width, tiled_height, 4)
        # # dim is (width, height) to match x, y convention
        # wp.launch(
        #     kernel=_convert_raw_rgb_tiled,
        #     dim=(tiled_width, tiled_height),
        #     inputs=[
        #         self._raw_output_rgb_buffer,
        #         self._output_data_buffers["rgba"],
        #         self._width,
        #         self._height,
        #         self._num_tiles_per_side,
        #     ],
        #     device=self._raw_output_rgb_buffer.device,
        # )

        # # Depth: float32 1D -> float32 tiled (tiled_width, tiled_height, 1)
        # # dim is (width, height) to match x, y convention
        # wp.launch(
        #     kernel=_convert_raw_depth_tiled,
        #     dim=(tiled_width, tiled_height),
        #     inputs=[
        #         self._raw_output_depth_buffer,
        #         self._output_data_buffers["depth"],
        #         self._width,
        #         self._height,
        #         self._num_tiles_per_side,
        #     ],
        #     device=self._raw_output_depth_buffer.device,
        # )

    def step(self):
        """Step the renderer."""
        pass

    def reset(self):
        """Reset the renderer."""
        pass

    def close(self):
        """Close the renderer."""
        pass
