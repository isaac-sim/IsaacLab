# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np
import scipy
import torch

from isaaclab.utils.assets import retrieve_file_path

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .visuotactile_sensor_cfg import GelSightRenderCfg


def compute_tactile_shear_image(
    tactile_normal_force: np.ndarray,
    tactile_shear_force: np.ndarray,
    normal_force_threshold: float = 0.00008,
    shear_force_threshold: float = 0.0005,
    resolution: int = 30,
) -> np.ndarray:
    """Visualize the tactile shear field.

    This function creates a visualization of tactile forces using arrows to represent shear forces
    and color coding to represent normal forces. The thresholds are used to normalize forces for
    visualization, chosen empirically to provide clear visual representation.

    Args:
        tactile_normal_force: Array of tactile normal forces. Shape: (H, W).
        tactile_shear_force: Array of tactile shear forces. Shape: (H, W, 2).
        normal_force_threshold: Threshold for normal force visualization. Defaults to 0.00008.
        shear_force_threshold: Threshold for shear force visualization. Defaults to 0.0005.
        resolution: Resolution for the visualization. Defaults to 30.

    Returns:
        Image visualizing the tactile shear forces. Shape: (H * resolution, W * resolution, 3).
    """
    nrows = tactile_normal_force.shape[0]
    ncols = tactile_normal_force.shape[1]

    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=float)

    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * resolution + resolution // 2
            loc0_y = col * resolution + resolution // 2
            loc1_x = loc0_x + tactile_shear_force[row, col][0] / shear_force_threshold * resolution
            loc1_y = loc0_y + tactile_shear_force[row, col][1] / shear_force_threshold * resolution
            color = (
                0.0,
                max(0.0, 1.0 - tactile_normal_force[row][col] / normal_force_threshold),
                min(1.0, tactile_normal_force[row][col] / normal_force_threshold),
            )

            cv2.arrowedLine(
                imgs_tactile, (int(loc0_y), int(loc0_x)), (int(loc1_y), int(loc1_x)), color, 6, tipLength=0.4
            )

    return imgs_tactile


def compute_penetration_depth(
    penetration_depth_img: np.ndarray, resolution: int = 5, depth_multiplier: float = 300.0
) -> np.ndarray:
    """Visualize the penetration depth.

    Args:
        penetration_depth_img: Image of penetration depth. Shape: (H, W).
        resolution: Resolution for the upsampling; each pixel expands to a (res x res) block. Defaults to 5.
        depth_multiplier: Multiplier for the depth values. Defaults to 300.0 (scales ~3.3mm to 1.0).
            (e.g. typical Gelsight sensors have maximum penetration depths < 2.5mm,
            see https://dspace.mit.edu/handle/1721.1/114627).

    Returns:
        Upsampled image visualizing the penetration depth. Shape: (H * resolution, W * resolution).
    """
    # penetration_depth_img_upsampled = penetration_depth.repeat(resolution, 0).repeat(resolution, 1)
    penetration_depth_img_upsampled = np.kron(penetration_depth_img, np.ones((resolution, resolution)))
    penetration_depth_img_upsampled = np.clip(penetration_depth_img_upsampled, 0.0, 1.0) * depth_multiplier
    return penetration_depth_img_upsampled


class GelsightRender:
    """Class to handle GelSight rendering using the Taxim example-based approach from :cite:t:`si2022taxim`.

    Reference:
        Si, Z., & Yuan, W. (2022). Taxim: An example-based simulation model for GelSight
        tactile sensors. IEEE Robotics and Automation Letters, 7(2), 2361-2368.
        https://arxiv.org/abs/2109.04027
    """

    def __init__(self, cfg: GelSightRenderCfg, device: str | torch.device):
        """Initialize the GelSight renderer.

        Args:
            cfg: Configuration object for the GelSight sensor.
            device: Device to use ('cpu' or 'cuda').

        Raises:
            ValueError: If :attr:`GelSightRenderCfg.mm_per_pixel` is zero or negative.
            FileNotFoundError: If render data files cannot be retrieved.
        """
        self.cfg = cfg
        self.device = device

        # Validate configuration parameters
        eps = 1e-9
        if self.cfg.mm_per_pixel < eps:
            raise ValueError(f"Input 'mm_per_pixel' must be positive (>= {eps}), got {self.cfg.mm_per_pixel}")

        # Retrieve render data files using the configured base path
        bg_path = self._get_render_data(self.cfg.sensor_data_dir_name, self.cfg.background_path)
        calib_path = self._get_render_data(self.cfg.sensor_data_dir_name, self.cfg.calib_path)

        if bg_path is None or calib_path is None:
            raise FileNotFoundError(
                "Failed to retrieve GelSight render data files. "
                f"Base path: {self.cfg.base_data_path or 'default (Isaac Lab Nucleus)'}, "
                f"Data dir: {self.cfg.sensor_data_dir_name}"
            )

        self.background = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)

        # Load calibration data directly
        calib_data = np.load(calib_path)
        calib_grad_r = calib_data["grad_r"]
        calib_grad_g = calib_data["grad_g"]
        calib_grad_b = calib_data["grad_b"]

        image_height = self.cfg.image_height
        image_width = self.cfg.image_width
        num_bins = self.cfg.num_bins
        [xx, yy] = np.meshgrid(range(image_width), range(image_height))
        xf = xx.flatten()
        yf = yy.flatten()
        self.A = np.array([xf * xf, yf * yf, xf * yf, xf, yf, np.ones(image_height * image_width)]).T

        binm = num_bins - 1
        self.x_binr = 0.5 * np.pi / binm  # x [0,pi/2]
        self.y_binr = 2 * np.pi / binm  # y [-pi, pi]

        kernel = self._get_filtering_kernel(kernel_size=5)
        self.kernel = torch.tensor(kernel, dtype=torch.float, device=self.device)

        self.calib_data_grad_r = torch.tensor(calib_grad_r, device=self.device)
        self.calib_data_grad_g = torch.tensor(calib_grad_g, device=self.device)
        self.calib_data_grad_b = torch.tensor(calib_grad_b, device=self.device)

        self.A_tensor = torch.tensor(self.A.reshape(image_height, image_width, 6), device=self.device).unsqueeze(0)
        self.background_tensor = torch.tensor(self.background, device=self.device)

        # Pre-allocate buffer for RGB output (will be resized if needed)
        self._sim_img_rgb_buffer = torch.empty((1, image_height, image_width, 3), device=self.device)

        logger.info("Gelsight renderer initialization done!")

    def render(self, height_map: torch.Tensor) -> torch.Tensor:
        """Render the height map using the GelSight sensor.

        Args:
            height_map: Input height map tensor. Shape is (N, H, W).

        Returns:
            Rendered image tensor. Shape is (N, H, W, 3).
        """
        height_map = height_map.clone()
        height_map[torch.abs(height_map) < 1e-6] = 0  # remove minor artifact
        height_map = height_map * -1000.0
        height_map /= self.cfg.mm_per_pixel

        height_map = self._gaussian_filtering(height_map.unsqueeze(-1), self.kernel).squeeze(-1)

        grad_mag, grad_dir = self._generate_normals(height_map)

        idx_x = torch.floor(grad_mag / self.x_binr).long()
        idx_y = torch.floor((grad_dir + np.pi) / self.y_binr).long()

        # Clamp indices to valid range to prevent out-of-bounds errors
        max_idx = self.cfg.num_bins - 1
        idx_x = torch.clamp(idx_x, 0, max_idx)
        idx_y = torch.clamp(idx_y, 0, max_idx)

        params_r = self.calib_data_grad_r[idx_x, idx_y, :]
        params_g = self.calib_data_grad_g[idx_x, idx_y, :]
        params_b = self.calib_data_grad_b[idx_x, idx_y, :]

        # Reuse pre-allocated buffer, resize if batch size changed
        target_shape = (*idx_x.shape, 3)
        if self._sim_img_rgb_buffer.shape != target_shape:
            self._sim_img_rgb_buffer = torch.empty(target_shape, device=self.device)
        sim_img_rgb = self._sim_img_rgb_buffer

        sim_img_rgb[..., 0] = torch.sum(self.A_tensor * params_r, dim=-1)  # R
        sim_img_rgb[..., 1] = torch.sum(self.A_tensor * params_g, dim=-1)  # G
        sim_img_rgb[..., 2] = torch.sum(self.A_tensor * params_b, dim=-1)  # B

        # write tactile image
        sim_img = sim_img_rgb + self.background_tensor  # /255.0
        sim_img = torch.clip(sim_img, 0, 255, out=sim_img).to(torch.uint8)
        return sim_img

    """
    Internal Helpers.
    """

    def _get_render_data(self, data_dir: str, file_name: str) -> str:
        """Gets the path for the GelSight render data file.

        Args:
            data_dir: The data directory name containing the render data.
            file_name: The specific file name to retrieve.

        Returns:
            The local path to the file.

        Raises:
            FileNotFoundError: If the file is not found locally or on Nucleus.
        """
        # Construct path using the configured base path
        file_path = os.path.join(self.cfg.base_data_path, data_dir, file_name)

        # Cache directory for downloads
        cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), data_dir)

        # Use retrieve_file_path to handle local/Nucleus paths and caching
        return retrieve_file_path(file_path, download_dir=cache_dir, force_download=False)

    def _generate_normals(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the gradient magnitude and direction of the height map.

        Args:
            img: Input height map tensor. Shape: (N, H, W).

        Returns:
            Tuple containing gradient magnitude tensor and gradient direction tensor. Shape: (N, H, W).
        """
        img_grad = torch.gradient(img, dim=(1, 2))
        dzdx, dzdy = img_grad

        grad_mag_orig = torch.sqrt(dzdx**2 + dzdy**2)
        grad_mag = torch.arctan(grad_mag_orig)  # seems that arctan is used as a squashing function
        grad_dir = torch.arctan2(dzdx, dzdy)
        grad_dir[grad_mag_orig == 0] = 0

        # handle edges
        grad_mag = torch.nn.functional.pad(grad_mag[:, 1:-1, 1:-1], pad=(1, 1, 1, 1))
        grad_dir = torch.nn.functional.pad(grad_dir[:, 1:-1, 1:-1], pad=(1, 1, 1, 1))

        return grad_mag, grad_dir

    def _get_filtering_kernel(self, kernel_size: int = 5) -> np.ndarray:
        """Create a Gaussian filtering kernel.

        For kernel derivation, see https://cecas.clemson.edu/~stb/ece847/internal/cvbook/ch03_filtering.pdf

        Args:
            kernel_size: Size of the kernel. Defaults to 5.

        Returns:
            Filtering kernel. Shape is (kernel_size, kernel_size).
        """
        filter_1D = scipy.special.binom(kernel_size - 1, np.arange(kernel_size))
        filter_1D /= filter_1D.sum()
        filter_1D = filter_1D[..., None]

        kernel = filter_1D @ filter_1D.T
        return kernel

    def _gaussian_filtering(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian filtering to the input image tensor.

        Args:
            img: Input image tensor. Shape is (N, H, W, 1).
            kernel: Filtering kernel tensor. Shape is (K, K).

        Returns:
            Filtered image tensor. Shape is (N, H, W, 1).
        """
        img_output = torch.nn.functional.conv2d(
            img.permute(0, 3, 1, 2), kernel.unsqueeze(0).unsqueeze(0), stride=1, padding="same"
        ).permute(0, 2, 3, 1)
        return img_output
