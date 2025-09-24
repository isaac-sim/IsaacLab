# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import scipy
import torch

import cv2

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

conf_r15 = {
    "data_dir": "gelsight_r15_data",
    "background_path": "bg.jpg",
    "calib_path": "polycalib.npz",
    "real_bg": "real_bg.npy",
    "h": 320,
    "w": 240,
    "numBins": 120,
    "pixmm": 0.0877,
}
conf_gs_mini = {
    "data_dir": "gs_mini_data",
    "background_path": "bg.jpg",
    "calib_path": "polycalib.npz",
    "real_bg": "real_bg.npy",
    "h": 240,
    "w": 320,
    "numBins": 120,
    "pixmm": 0.065,
}
conf_options = {
    "gelsight_r15": conf_r15,
    "gs_mini": conf_gs_mini,
}


def padding(img):
    """
    Apply symmetric padding to the input image.

    Parameters:
    img (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Padded image.
    """
    if len(img.shape) == 2:
        return np.pad(img, ((1, 1), (1, 1)), "symmetric")
    elif len(img.shape) == 3:
        return np.pad(img, ((1, 1), (1, 1), (0, 0)), "symmetric")


def compute_image_gradient(image):
    """
    Compute the gradient of an image.

    Parameters:
    image (numpy.ndarray): Input image.

    Returns:
    tuple: Gradients in x and y directions.
    """
    dzdx, dzdy = np.gradient(image)
    dzdx = dzdx[1:-1, 1:-1]  # remove boundary values. Note edge_order=1 is used to calculate boundary values
    dzdy = dzdy[1:-1, 1:-1]
    return dzdx, dzdy


def generate_normals(height_map):
    """
    Generate the gradient magnitude and direction of the height map.

    Parameters:
    height_map (numpy.ndarray): Input height map.

    Returns:
    tuple: Gradient magnitude, gradient direction, and None.
    """
    dzdx, dzdy = compute_image_gradient(height_map)

    grad_mag_orig = np.sqrt(dzdx**2 + dzdy**2)
    grad_mag = np.arctan(grad_mag_orig)  # seems that arctan is used as a squashing function
    grad_dir = np.arctan2(dzdx, dzdy)
    grad_dir[grad_mag_orig == 0] = 0

    grad_mag = padding(grad_mag)
    grad_dir = padding(grad_dir)
    return grad_mag, grad_dir, None


def generate_normals_tensor(img_tensor):
    """
    Generate the gradient magnitude and direction of the height map using tensors.

    Parameters:
    img_tensor (torch.Tensor): Input height map tensor.

    Returns:
    tuple: Gradient magnitude tensor, gradient direction tensor, and None.
    """
    img_grad_tensor = torch.gradient(img_tensor, dim=(1, 2))
    dzdx, dzdy = img_grad_tensor

    grad_mag_orig_tensor = torch.sqrt(dzdx**2 + dzdy**2)
    grad_mag_tensor = torch.arctan(grad_mag_orig_tensor)  # seems that arctan is used as a squashing function
    grad_dir_tensor = torch.arctan2(dzdx, dzdy)
    grad_dir_tensor[grad_mag_orig_tensor == 0] = 0

    # handle edges
    grad_mag_tensor = torch.nn.functional.pad(grad_mag_tensor[:, 1:-1, 1:-1], pad=(1, 1, 1, 1))
    grad_dir_tensor = torch.nn.functional.pad(grad_dir_tensor[:, 1:-1, 1:-1], pad=(1, 1, 1, 1))

    return grad_mag_tensor, grad_dir_tensor, None


def get_filtering_kernel(kernel_sz=5):
    """
    Create a Gaussian filtering kernel.
    # For kernel derivation, see https://cecas.clemson.edu/~stb/ece847/internal/cvbook/ch03_filtering.pdf

    Parameters:
    kernel_sz (int): Size of the kernel.

    Returns:
    numpy.ndarray: Filtering kernel.
    """
    filter_1D = scipy.special.binom(kernel_sz - 1, np.arange(kernel_sz))
    filter_1D /= filter_1D.sum()
    filter_1D = filter_1D[..., None]

    kernel = filter_1D @ filter_1D.T
    return kernel


def gaussian_filtering(img_tensor, kernel_tensor):
    """
    Apply Gaussian filtering to the input image tensor.

    Parameters:
    img_tensor (torch.Tensor): Input image tensor.
    kernel_tensor (torch.Tensor): Filtering kernel tensor.

    Returns:
    torch.Tensor: Filtered image tensor.
    """
    img_tensor_output = torch.nn.functional.conv2d(
        img_tensor.permute(0, 3, 1, 2), kernel_tensor.unsqueeze(0).unsqueeze(0), stride=1, padding="same"
    ).permute(0, 2, 3, 1)
    return img_tensor_output


def get_gs_render_data(data_dir: str, file_name: str) -> str | None:
    """Gets the path for the GelSight render data file.

    If the data file is not cached locally then the file is downloaded from
    the Isaac Lab Nucleus directory. The cached path is then returned.

    Args:
        data_dir: The data directory name containing the render data.
        file_name: The specific file name to retrieve.

    Returns:
        The local path to the downloaded/cached file, or None if unavailable.
    """
    ov_path = os.path.join(ISAACLAB_NUCLEUS_DIR, "TacSL", data_dir, file_name)
    download_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), data_dir)
    os.makedirs(download_dir, exist_ok=True)
    download_path = os.path.join(download_dir, file_name)

    if not os.path.exists(download_path):
        print(f"Fetching gs render data : {ov_path}")
        try:
            download_path = retrieve_file_path(ov_path, download_dir)
        except Exception:
            print("A gs render data is currently unavailable for this task.")
            return None
    else:
        print("Using pre-fetched gs render data: ", download_path)
    return download_path


class CalibData:
    """
    Class to handle calibration data.
    """

    def __init__(self, dataPath):
        """
        Initialize the calibration data.

        Parameters:
        dataPath (str): Path to the calibration data file.
        """
        self.dataPath = dataPath
        data = np.load(dataPath)

        self.numBins = data["bins"]
        self.grad_r = data["grad_r"]
        self.grad_g = data["grad_g"]
        self.grad_b = data["grad_b"]


class gelsightRender:
    """
    Class to handle GelSight rendering using the Taxim example-based approach.
        Ref: https://arxiv.org/abs/2109.04027
    """

    def __init__(self, sensor_name, device):
        """
        Initialize the GelSight renderer.

        Parameters:
        sensor_name (str): Name of the sensor.
        device (str): Device to use ('cpu' or 'cuda').
        """

        self.sensor_name = sensor_name
        self.device = device
        self.conf = conf_options[self.sensor_name]

        bg_path = get_gs_render_data(self.conf["data_dir"], self.conf["background_path"])
        calib_path = get_gs_render_data(self.conf["data_dir"], self.conf["calib_path"])

        self.background = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)

        self.calib_data = CalibData(calib_path)
        h, w = self.conf["h"], self.conf["w"]
        bins = self.conf["numBins"]
        [xx, yy] = np.meshgrid(range(w), range(h))
        xf = xx.flatten()
        yf = yy.flatten()
        self.A = np.array([xf * xf, yf * yf, xf * yf, xf, yf, np.ones(h * w)]).T

        binm = bins - 1
        self.x_binr = 0.5 * np.pi / binm  # x [0,pi/2]
        self.y_binr = 2 * np.pi / binm  # y [-pi, pi]

        kernel = get_filtering_kernel(kernel_sz=5)
        self.kernel = torch.tensor(kernel, dtype=torch.float, device=self.device)

        self.calib_data_grad_r_tensor = torch.tensor(self.calib_data.grad_r, device=self.device)
        self.calib_data_grad_g_tensor = torch.tensor(self.calib_data.grad_g, device=self.device)
        self.calib_data_grad_b_tensor = torch.tensor(self.calib_data.grad_b, device=self.device)

        self.A_tensor = torch.tensor(self.A.reshape(h, w, 6), device=self.device).unsqueeze(0)
        self.background_tensor = torch.tensor(self.background, device=self.device)
        print("Gelsight initialization done!")

    def render(self, heightMap):
        """
        Render the height map using the GelSight sensor.

        Parameters:
        heightMap (numpy.ndarray): Input height map.

        Returns:
        numpy.ndarray: Rendered image.
        """
        # print("gelsight render")
        height_map = heightMap.copy()
        height_map[np.abs(height_map) < 1e-6] = 0  # remove minor artifact
        height_map = height_map * -1000.0
        height_map /= self.conf["pixmm"]

        height_map = cv2.GaussianBlur(height_map.astype(np.float32), (5, 5), 0)
        grad_mag, grad_dir, _ = generate_normals(height_map)

        h, w = self.conf["h"], self.conf["w"]
        sim_img_rgb = np.zeros((h, w, 3))
        idx_x = np.floor(grad_mag / self.x_binr).astype("int")
        idx_y = np.floor((grad_dir + np.pi) / self.y_binr).astype("int")

        params_r = self.calib_data.grad_r[idx_x, idx_y, :]
        params_g = self.calib_data.grad_g[idx_x, idx_y, :]
        params_b = self.calib_data.grad_b[idx_x, idx_y, :]

        sim_img_rgb[:, :, 0] = np.sum(self.A.reshape(*grad_mag.shape, 6) * params_r, axis=-1)  # R
        sim_img_rgb[:, :, 1] = np.sum(self.A.reshape(*grad_mag.shape, 6) * params_g, axis=-1)  # G
        sim_img_rgb[:, :, 2] = np.sum(self.A.reshape(*grad_mag.shape, 6) * params_b, axis=-1)  # B

        # write tactile image
        sim_img = sim_img_rgb + self.background  # /255.0
        return sim_img

    def render_tensorized(self, heightMap):
        """
        Render the height map using the GelSight sensor (tensorized version).

        Parameters:
        heightMap (torch.Tensor): Input height map tensor.

        Returns:
        torch.Tensor: Rendered image tensor.
        """
        height_map = heightMap.clone()
        height_map[torch.abs(height_map) < 1e-6] = 0  # remove minor artifact
        height_map = height_map * -1000.0
        height_map /= self.conf["pixmm"]

        height_map = gaussian_filtering(height_map.unsqueeze(-1), self.kernel).squeeze(-1)

        grad_mag_tensor, grad_dir_tensor, _ = generate_normals_tensor(height_map)

        idx_x_tensor = torch.floor(grad_mag_tensor / self.x_binr).long()
        idx_y_tensor = torch.floor((grad_dir_tensor + np.pi) / self.y_binr).long()

        params_r_tensor = self.calib_data_grad_r_tensor[idx_x_tensor, idx_y_tensor, :]
        params_g_tensor = self.calib_data_grad_g_tensor[idx_x_tensor, idx_y_tensor, :]
        params_b_tensor = self.calib_data_grad_b_tensor[idx_x_tensor, idx_y_tensor, :]

        sim_img_rgb_tensor = torch.zeros((*idx_x_tensor.shape, 3), device=self.device)  # TODO: move to init
        sim_img_rgb_tensor[..., 0] = torch.sum(self.A_tensor * params_r_tensor, axis=-1)  # R
        sim_img_rgb_tensor[..., 1] = torch.sum(self.A_tensor * params_g_tensor, axis=-1)  # G
        sim_img_rgb_tensor[..., 2] = torch.sum(self.A_tensor * params_b_tensor, axis=-1)  # B

        # write tactile image
        sim_img = sim_img_rgb_tensor + self.background_tensor  # /255.0
        sim_img = torch.clip(sim_img, 0, 255, out=sim_img).to(torch.uint8)
        return sim_img
