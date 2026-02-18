# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL Visualizer implementation."""

import math

import torch
import warp as wp
from newton.sensors import SensorTiledCamera

from isaaclab.managers.newton_manager import NewtonManager
from isaaclab.utils.math import convert_camera_frame_orientation_convention
from isaaclab.utils.timer import Timer

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


@wp.kernel
def _copy_depth_with_channel(
    src: wp.array(dtype=wp.float32, ndim=3),  # shape: (num_envs, H, W)
    dst: wp.array(dtype=wp.float32, ndim=4),  # shape: (num_envs, H, W, 1)
):
    """Copy depth values and add channel dimension."""
    env_id, y, x = wp.tid()
    dst[env_id, y, x, 0] = src[env_id, y, x]


class NewtonWarpRenderer(RendererBase):
    """Newton Warp Renderer implementation."""

    _model = None

    # tiled camerae sensor from warp trace
    _tiled_camera_sensor = None

    def __init__(self, cfg: NewtonWarpRendererCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self._render_call_count = 0
        self._last_num_envs = getattr(self, "_num_envs", 1)  # updated each render(); used for save_image tiled grid

        # Create save directory (will be cleaned up on shutdown)
        import os
        import shutil

        self._save_dir = "/tmp/newton_renders"
        if os.path.exists(self._save_dir):
            shutil.rmtree(self._save_dir)
        os.makedirs(self._save_dir, exist_ok=True)

    def initialize(self):
        """Initialize the renderer."""
        import sys

        print(
            "[NewtonWarpRenderer] initialize() called — Newton Warp renderer active (debug + timing enabled).",
            flush=True,
        )
        sys.stdout.flush()
        self._model = NewtonManager.get_model()

        # Create tiled camera sensor. With one Newton model (one world) and num_cameras=num_envs,
        # each tile is one camera view of the same full scene, so each tile shows all envs.
        # To get one env per tile (like Newton-Warp reference), the pipeline would need
        # num_worlds=num_envs and num_cameras=1 (one camera per world); that requires the
        # Newton model to expose per-env worlds (e.g. replicated scenes).
        self._tiled_camera_sensor = SensorTiledCamera(
            model=self._model,
            num_cameras=self._num_envs,
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

        # Raw buffers from the tiled camera sensor; output buffers are views set each frame in _copy_outputs_to_buffers()
        self._raw_output_rgb_buffer = self._tiled_camera_sensor.create_color_image_output()
        self._raw_output_depth_buffer = self._tiled_camera_sensor.create_depth_image_output()

    def _prepare_camera_transforms(
        self, camera_positions: torch.Tensor, camera_orientations: torch.Tensor, intrinsic_matrices: torch.Tensor
    ):
        """Convert torch camera data to Warp camera_transforms (for timing: this is pre-kernel setup)."""
        if self._camera_rays is None:
            self.set_camera_rays_from_intrinsics(intrinsic_matrices)
        num_envs = camera_positions.shape[0]
        camera_positions_wp = wp.from_torch(camera_positions.contiguous(), dtype=wp.vec3)
        camera_quats_converted = convert_camera_frame_orientation_convention(
            camera_orientations, origin="world", target="opengl"
        )
        camera_orientations_wp = wp.from_torch(camera_quats_converted, dtype=wp.quat)
        camera_transforms = wp.empty((num_envs, 1), dtype=wp.transformf, device=camera_positions_wp.device)
        wp.launch(
            kernel=_create_camera_transforms_kernel,
            dim=num_envs,
            inputs=[camera_positions_wp, camera_orientations_wp, camera_transforms],
            device=camera_positions_wp.device,
        )
        return camera_transforms

    def _render_warp_kernel_only(self, camera_transforms: wp.array):
        """Run only SensorTiledCamera.render() (Warp ray trace). Use this for apples-to-apples timing vs Newton-Warp."""
        self._tiled_camera_sensor.render(
            state=NewtonManager.get_state_0(),
            camera_transforms=camera_transforms,
            camera_rays=self._camera_rays,
            color_image=self._raw_output_rgb_buffer,
            depth_image=self._raw_output_depth_buffer,
        )

    def _copy_outputs_to_buffers(self, num_envs: int):
        """Copy raw sensor output into output buffers using views (zero-copy; avoids per-env wp.copy)."""
        rgb_reshaped = self._raw_output_rgb_buffer.reshape((num_envs, self._height * self._width))
        rgba_uint8 = wp.array(
            ptr=rgb_reshaped.ptr,
            shape=(num_envs, self._height, self._width, 4),
            dtype=wp.uint8,
            device=rgb_reshaped.device,
        )
        self._output_data_buffers["rgba"] = rgba_uint8
        self._output_data_buffers["rgb"] = rgba_uint8[:, :, :, :3]
        self._output_data_buffers["depth"] = self._raw_output_depth_buffer.reshape(
            (num_envs, self._height, self._width, 1)
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
        num_envs = camera_positions.shape[0]

        # Full render timer (apples-to-apples with Newton+Warp: prep + kernel + buffer copy)
        with Timer(name="newton_warp_render_full", msg="Newton Warp full render took"):
            with Timer(name="newton_warp_prep", msg="Newton Warp prep took"):
                camera_transforms = self._prepare_camera_transforms(
                    camera_positions, camera_orientations, intrinsic_matrices
                )
            with Timer(name="newton_warp_kernel_only", msg="Newton Warp kernel only took"):
                self._render_warp_kernel_only(camera_transforms)
            with Timer(name="newton_warp_copy_buffers", msg="Newton Warp copy buffers took"):
                self._copy_outputs_to_buffers(num_envs)

        self._last_num_envs = num_envs  # for save_image tiled grid (buffer.numpy() shape may not match)
        # Debug save every 50 frames (outside timed region)
        self._render_call_count += 1
        if self._render_call_count % 50 == 0:
            import os
            import sys

            frame_dir = os.path.join(self._save_dir, f"frame_{self._render_call_count:06d}")
            os.makedirs(frame_dir, exist_ok=True)
            tiled_rgb = os.path.join(frame_dir, "all_envs_tiled_rgb.png")
            self.save_image(tiled_rgb, env_index=None, data_type="rgb")
            print(f"[NewtonWarpRenderer] Saved tiled RGB → {frame_dir}/", flush=True)
            try:
                for timer_name in (
                    "newton_warp_render_full",
                    "newton_warp_prep",
                    "newton_warp_kernel_only",
                    "newton_warp_copy_buffers",
                ):
                    stats = Timer.get_timer_statistics(timer_name)
                    print(
                        f"[NewtonWarpRenderer] {timer_name}: mean={stats['mean']:.6f}s std={stats['std']:.6f}s n={stats['n']}",
                        flush=True,
                    )
                for timer_name in ("newton_state_sync_usdrt", "newton_state_sync_tensors"):
                    try:
                        stats = Timer.get_timer_statistics(timer_name)
                        print(
                            f"[NewtonWarpRenderer] {timer_name}: mean={stats['mean']:.6f}s std={stats['std']:.6f}s n={stats['n']}",
                            flush=True,
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            sys.stdout.flush()

    def save_image(self, filename: str, env_index: int | None = 0, data_type: str = "rgb"):
        """Save a single environment or a tiled grid of environments to disk.

        Args:
            filename: Path to save the image (should end with .png).
            env_index: Environment index to save, or None for tiled grid of all envs.
            data_type: Which data to save - "rgb", "rgba", or "depth". Default: "rgb".
        """
        import numpy as np
        from PIL import Image

        if data_type == "rgb" and "rgb" in self._output_data_buffers:
            buffer = self._output_data_buffers["rgb"]
            mode = "RGB"
        elif data_type == "rgba" and "rgba" in self._output_data_buffers:
            buffer = self._output_data_buffers["rgba"]
            mode = "RGBA"
        elif data_type == "depth" and "depth" in self._output_data_buffers:
            buffer = self._output_data_buffers["depth"]
            mode = "L"
        else:
            raise ValueError(f"Data type '{data_type}' not available in output buffers.")

        buffer_np = buffer.numpy()
        num_envs_from_buffer = buffer_np.shape[0] if len(buffer_np.shape) >= 4 else 1
        num_envs_for_tile = getattr(self, "_last_num_envs", None)
        if num_envs_for_tile is None:
            num_envs_for_tile = num_envs_from_buffer
        n_expected = int(num_envs_for_tile)
        channels = 1 if data_type == "depth" else (4 if data_type == "rgba" else 3)
        expected_size = n_expected * self._height * self._width * channels
        if buffer_np.size == expected_size and num_envs_from_buffer != n_expected and buffer_np.size > 0:
            try:
                buffer_np = buffer_np.reshape((n_expected, self._height, self._width, channels))
                num_envs_from_buffer = n_expected
            except (ValueError, AttributeError):
                pass

        if env_index is None:
            num_envs = min(int(num_envs_for_tile), num_envs_from_buffer)
            tiles_per_side = int(np.ceil(np.sqrt(num_envs)))
            tiled_height = tiles_per_side * self._height
            tiled_width = tiles_per_side * self._width

            if data_type == "depth":
                tiled_image = np.zeros((tiled_height, tiled_width), dtype=np.uint8)
                for idx in range(num_envs):
                    tile_y = idx // tiles_per_side
                    tile_x = idx % tiles_per_side
                    y_start = tile_y * self._height
                    y_end = y_start + self._height
                    x_start = tile_x * self._width
                    x_end = x_start + self._width
                    depth_data = buffer_np[idx, :, :, 0]
                    d_min, d_max = depth_data.min(), depth_data.max()
                    if d_max > d_min:
                        depth_vis = ((depth_data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                    else:
                        depth_vis = np.zeros_like(depth_data, dtype=np.uint8)
                    tiled_image[y_start:y_end, x_start:x_end] = depth_vis
            else:
                channels = 3 if mode == "RGB" else 4
                tiled_image = np.zeros((tiled_height, tiled_width, channels), dtype=np.uint8)
                for idx in range(num_envs):
                    tile_y = idx // tiles_per_side
                    tile_x = idx % tiles_per_side
                    y_start = tile_y * self._height
                    y_end = y_start + self._height
                    x_start = tile_x * self._width
                    x_end = x_start + self._width
                    tiled_image[y_start:y_end, x_start:x_end] = buffer_np[idx]

            img = Image.fromarray(tiled_image, mode=mode)
            img.save(filename)
            print(f"[NewtonWarpRenderer] Saved tiled {data_type} image: {filename}", flush=True)
        else:
            if data_type == "depth":
                depth_data = buffer_np[env_index, :, :, 0]
                d_min, d_max = depth_data.min(), depth_data.max()
                if d_max > d_min:
                    img_data = ((depth_data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    img_data = np.zeros_like(depth_data, dtype=np.uint8)
            else:
                img_data = buffer_np[env_index]
            img = Image.fromarray(img_data, mode=mode)
            img.save(filename)
            print(f"[NewtonWarpRenderer] Saved env {env_index} {data_type} image: {filename}", flush=True)

    def step(self):
        """Step the renderer."""
        pass

    def reset(self):
        """Reset the renderer."""
        pass

    def close(self):
        """Close the renderer."""
        pass
