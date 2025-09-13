# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import cv2


def visualize_tactile_shear_image(
    tactile_normal_force,
    tactile_shear_force,
    normal_force_threshold=0.00008,
    shear_force_threshold=0.0005,
    resolution=30,
):
    """
    Visualize the tactile shear field.

    Args:
        tactile_normal_force (np.ndarray): Array of tactile normal forces.
        tactile_shear_force (np.ndarray): Array of tactile shear forces.
        normal_force_threshold (float): Threshold for normal force visualization.
        shear_force_threshold (float): Threshold for shear force visualization.
        resolution (int): Resolution for the visualization.

    Returns:
        np.ndarray: Image visualizing the tactile shear forces.
    """
    nrows = tactile_normal_force.shape[0]
    ncols = tactile_normal_force.shape[1]

    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=float)

    # print('(min, max) tactile normal force: ', np.min(tactile_normal_force), np.max(tactile_normal_force))
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


def visualize_penetration_depth(penetration_depth_img, resolution=5, depth_multiplier=300.0):
    """
    Visualize the penetration depth.

    Args:
        penetration_depth_img (np.ndarray): Image of penetration depth.
        resolution (int): Resolution for the upsampling.
        depth_multiplier (float): Multiplier for the depth values.

    Returns:
        np.ndarray: Upsampled image visualizing the penetration depth.
    """
    # penetration_depth_img_upsampled = penetration_depth.repeat(resolution, 0).repeat(resolution, 1)
    print("penetration_depth_img: ", np.max(penetration_depth_img))
    penetration_depth_img_upsampled = np.kron(penetration_depth_img, np.ones((resolution, resolution)))
    penetration_depth_img_upsampled = np.clip(penetration_depth_img_upsampled, 0.0, 1.0) * depth_multiplier
    return penetration_depth_img_upsampled
