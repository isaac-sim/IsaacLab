# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# isort: off
import warnings
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
# isort: on

import numpy as np
import scipy.interpolate as interpolate


@pytest.mark.isaacsim_ci
def test_interpolation():
    """Test scipy interpolation 2D method."""
    # parameters
    size = (10.0, 12.0)
    horizontal_scale = 0.1
    vertical_scale = 0.005
    downsampled_scale = 0.2
    noise_range = (-0.02, 0.1)
    noise_step = 0.02
    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(size[0] / horizontal_scale)
    length_pixels = int(size[1] / horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(size[0] / downsampled_scale)
    length_downsampled = int(size[1] / downsampled_scale)
    # -- height
    height_min = int(noise_range[0] / vertical_scale)
    height_max = int(noise_range[1] / vertical_scale)
    height_step = int(noise_step / vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, size[0] * horizontal_scale, width_downsampled)
    y = np.linspace(0, size[1] * horizontal_scale, length_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, size[0] * horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, size[1] * horizontal_scale, length_pixels)
    # -- method 1: RegularGridInterpolator (replacing deprecated interp2d)
    func_RegularGridInterpolator = interpolate.RegularGridInterpolator((x, y), height_field_downsampled, method="cubic")
    xx_upsampled, yy_upsampled = np.meshgrid(x_upsampled, y_upsampled, indexing="ij", sparse=True)
    z_upsampled_RegularGridInterpolator = func_RegularGridInterpolator((xx_upsampled, yy_upsampled))
    # -- method 2: RectBivariateSpline (alternate to interp2d)
    func_RectBiVariate = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    z_upsampled_RectBivariant = func_RectBiVariate(x_upsampled, y_upsampled)

    # check if the interpolated height field is the same as the sampled height field
    np.testing.assert_allclose(z_upsampled_RegularGridInterpolator, z_upsampled_RectBivariant, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(z_upsampled_RectBivariant, z_upsampled_RegularGridInterpolator, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(
        z_upsampled_RegularGridInterpolator, z_upsampled_RegularGridInterpolator, atol=1e-2, rtol=1e-2
    )
