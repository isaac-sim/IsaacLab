# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni


def convert_camera_intrinsics_to_usd(
    intrinsic_matrix: list[float], width: int, height: int, focal_length: float | None = None
) -> dict:
    """Creates USD camera properties from camera intrinsics and resolution.

    Args:
        intrinsic_matrix: Intrinsic matrix of the camera in row-major format.
            The matrix is defined as [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]. Shape is (9,).
        width: Width of the image (in pixels).
        height: Height of the image (in pixels).
        focal_length: Perspective focal length (in cm) used to calculate pixel size. Defaults to None. If None
                focal_length will be calculated 1 / width.

    Returns:
        A dictionary of USD camera parameters for focal_length, horizontal_aperture, vertical_aperture,
            horizontal_aperture_offset, and vertical_aperture_offset.
    """
    usd_params = dict

    # extract parameters from matrix
    f_x = intrinsic_matrix[0]
    f_y = intrinsic_matrix[4]
    c_x = intrinsic_matrix[2]
    c_y = intrinsic_matrix[5]

    # warn about non-square pixels
    if abs(f_x - f_y) > 1e-4:
        omni.log.warn("Camera non square pixels are not supported by Omniverse. The average of f_x and f_y are used.")

    # warn about aperture offsets
    if abs((c_x - float(width) / 2) > 1e-4 or (c_y - float(height) / 2) > 1e-4):
        omni.log.warn(
            "Camera aperture offsets are not supported by Omniverse. c_x and c_y will be half of width and height"
        )

    # calculate usd camera parameters
    # pixel_size does not need to be exact it is just used for consistent sizing of aperture and focal_length
    # https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html#calibrated-camera-sensors
    if focal_length is None:
        pixel_size = 1 / float(width)
    else:
        pixel_size = focal_length / ((f_x + f_y) / 2)

    usd_params = {
        "horizontal_aperture": pixel_size * float(width),
        "vertical_aperture": pixel_size * float(height),
        "focal_length": pixel_size * (f_x + f_y) / 2,  # The focal length in mm
        "horizontal_aperture_offset": 0.0,
        "vertical_aperture_offset": 0.0,
    }

    return usd_params
