# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

from newton.sensors import TiledCameraSensor
from isaaclab.sim._impl.newton_manager import NewtonManager

from .newton_warp_renderer_cfg import NewtonWarpRendererCfg
from .renderer import RendererBase

import math
import torch
import warp as wp


@wp.kernel
def _create_camera_transforms_kernel(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
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
    transforms[i, 0] = wp.transformf(positions[i], orientations[i])


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
            num_cameras=1, # 1 because model already has 
            width=self._width,
            height=self._height,
            options=TiledCameraSensor.Options(
                default_light=True, default_light_shadows=True, colors_per_shape=True, checkerboard_texture=True
            )
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
        self._data_types = ["rgb", "depth"]
        self._output_data_buffers["rgb"] = self._tiled_camera_sensor.create_color_image_output()
        self._output_data_buffers["depth"] = self._tiled_camera_sensor.create_depth_image_output()

    def render(self, camera_positions: torch.Tensor, camera_orientations: torch.Tensor, intrinsic_matrices: torch.Tensor):
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
        
        # Quaternions: need to reorder from (w,x,y,z) to (x,y,z,w) for warp
        # Create a copy with reordered quaternion components
        camera_quats_reordered = torch.stack([
            camera_orientations[:, 1],  # x
            camera_orientations[:, 2],  # y
            camera_orientations[:, 3],  # z
            camera_orientations[:, 0],  # w
        ], dim=1).contiguous()  # Shape: (num_cameras, 4)

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
            NewtonManager.get_state_1(),  # Use current physics state
            camera_transforms,
            self._camera_rays,
            self._output_data_buffers["rgb"],
            self._output_data_buffers["depth"],
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