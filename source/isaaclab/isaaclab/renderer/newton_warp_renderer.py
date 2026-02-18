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
        positions: Array of camera positions, shape (num_envs,)
        orientations: Array of camera orientations, shape (num_envs,)
        transforms: Output array of camera transforms, shape (num_envs, num_cameras)
    Note:
        Currently only fills the first camera slot (index 0) as the implementation
        only supports 1 camera per environment.
    """
    i = wp.tid()
    transforms[i, 0] = wp.transformf(positions[i], orientations[i])


@wp.kernel
def _transpose_camera_transforms_kernel(
    input_transforms: wp.array(dtype=wp.transformf, ndim=2),
    output_transforms: wp.array(dtype=wp.transformf, ndim=2),
):
    """Kernel to transpose camera transforms from (num_envs, 1) to (1, num_envs).

    Args:
        input_transforms: Input array of shape (num_envs, 1)
        output_transforms: Output array of shape (1, num_envs)
    """
    i = wp.tid()
    output_transforms[0, i] = input_transforms[i, 0]


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
            options=SensorTiledCamera.Options(colors_per_shape=True),
        )

        # Note: camera rays will be computed when we have access to TiledCamera
        # for now use default 45 degree FOV
        self._camera_rays = None

        # Initialize output buffers
        self._initialize_output()

    def set_camera_rays_from_intrinsics(self, intrinsic_matrices: torch.Tensor):
        """Set camera FOV from intrinsic matrices.

        Args:
            intrinsic_matrices: Camera intrinsic matrices of shape (num_envs, 3, 3)
                             Format: [[f_x,   0, c_x],
                                     [  0, f_y, c_y],
                                     [  0,   0,   1]]
        """
        # Note: intrinsic_matrices has shape (num_envs, 3, 3), but we only have 1 camera
        # Use the first intrinsic matrix (assuming all cameras have the same intrinsics)
        f_y = intrinsic_matrices[0, 1, 1].item()  # Single camera's vertical focal length in pixels
        fov_radians = 2.0 * math.atan(self._height / (2.0 * f_y))
        self._camera_rays = self._tiled_camera_sensor.compute_pinhole_camera_rays(
            self._width, self._height, fov_radians
        )

    def _initialize_output(self):
        """Initialize the output of the renderer."""
        self._data_types = ["rgba", "rgb", "depth"]
        self._num_tiles_per_side = math.ceil(math.sqrt(self._num_envs))

        # Raw buffer to hold data from the tiled camera sensor
        self._raw_output_rgb_buffer = self._tiled_camera_sensor.create_color_image_output(
            self._width, self._height, self._num_cameras
        )
        self._raw_output_depth_buffer = self._tiled_camera_sensor.create_depth_image_output(
            self._width, self._height, self._num_cameras
        )

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
        camera_positions_wp = wp.from_torch(camera_positions.contiguous(), dtype=wp.vec3)
        camera_quats_converted = convert_camera_frame_orientation_convention(
            camera_orientations, origin="world", target="opengl"
        )
        camera_orientations_wp = wp.from_torch(camera_quats_converted, dtype=wp.quat)

        # Create camera transforms array
        # Format: wp.array of shape (num_cameras, num_worlds), dtype=wp.transformf
        # Note: SensorTiledCamera expects (num_cameras, num_worlds), not (num_worlds, num_cameras)
        # First create transforms in shape (num_envs, num_cameras) for the kernel
        temp_transforms = wp.empty(
            (num_envs, self._num_cameras), dtype=wp.transformf, device=camera_positions_wp.device
        )
        wp.launch(
            kernel=_create_camera_transforms_kernel,
            dim=num_envs,
            inputs=[camera_positions_wp, camera_orientations_wp, temp_transforms],
            device=camera_positions_wp.device,
        )
        # Transpose from (num_envs, num_cameras) to (num_cameras, num_envs) using a kernel
        camera_transforms = wp.empty(
            (self._num_cameras, num_envs), dtype=wp.transformf, device=camera_positions_wp.device
        )
        wp.launch(
            kernel=_transpose_camera_transforms_kernel,
            dim=num_envs,
            inputs=[temp_transforms, camera_transforms],
            device=camera_positions_wp.device,
        )

        # Render using SensorTiledCamera
        self._tiled_camera_sensor.render(
            state=NewtonManager.get_state_0(),
            camera_transforms=camera_transforms,
            camera_rays=self._camera_rays,
            color_image=self._raw_output_rgb_buffer,
            depth_image=self._raw_output_depth_buffer,
        )

        # Convert uint32 to uint8 RGBA
        reshape_rgba = self._raw_output_rgb_buffer.reshape((self._num_envs, self._height, self._width))
        self._output_data_buffers["rgba"] = wp.array(
            ptr=reshape_rgba.ptr, shape=(*reshape_rgba.shape, 4), dtype=wp.uint8
        )

        self._output_data_buffers["rgb"] = self._output_data_buffers["rgba"][:, :, :, :3]

        # Reshape depth buffer: (num_envs, num_cameras, 1, width*height) -> (num_envs, height, width, 1)
        # Note: Current implementation only supports 1 camera per environment.
        self._output_data_buffers["depth"] = self._raw_output_depth_buffer.reshape(
            (self._num_envs, self._height, self._width, 1)
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
