# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipy.spatial.transform as tf
import trimesh

"""
Primitive functions to generate meshes.
"""


def make_plane(size: tuple[float, float], height: float, center_zero: bool = True) -> trimesh.Trimesh:
    """Generate a plane mesh.

    If :obj:`center_zero` is True, the origin is at center of the plane mesh i.e. the mesh extends from
    :math:`(-size[0] / 2, -size[1] / 2, 0)` to :math:`(size[0] / 2, size[1] / 2, height)`.
    Otherwise, the origin is :math:`(size[0] / 2, size[1] / 2)` and the mesh extends from
    :math:`(0, 0, 0)` to :math:`(size[0], size[1], height)`.

    Args:
        size: The length (along x) and width (along y) of the terrain (in m).
        height: The height of the plane (in m).
        center_zero: Whether the 2D origin of the plane is set to the center of mesh.
            Defaults to True.

    Returns:
        A trimesh.Trimesh objects for the plane.
    """
    # compute the vertices of the terrain
    x0 = [size[0], size[1], height]
    x1 = [size[0], 0.0, height]
    x2 = [0.0, size[1], height]
    x3 = [0.0, 0.0, height]
    # generate the tri-mesh with two triangles
    vertices = np.array([x0, x1, x2, x3])
    faces = np.array([[1, 0, 2], [2, 3, 1]])
    plane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # center the plane at the origin
    if center_zero:
        plane_mesh.apply_translation(-np.array([size[0] / 2.0, size[1] / 2.0, 0.0]))
    # return the tri-mesh and the position
    return plane_mesh


def make_border(
    size: tuple[float, float], inner_size: tuple[float, float], height: float, position: tuple[float, float, float]
) -> list[trimesh.Trimesh]:
    """Generate meshes for a rectangular border with a hole in the middle.

    .. code:: text

        +---------------------+
        |#####################|
        |##+---------------+##|
        |##|               |##|
        |##|               |##| length
        |##|               |##| (y-axis)
        |##|               |##|
        |##+---------------+##|
        |#####################|
        +---------------------+
              width (x-axis)

    Args:
        size: The length (along x) and width (along y) of the terrain (in m).
        inner_size: The inner length (along x) and width (along y) of the hole (in m).
        height: The height of the border (in m).
        position: The center of the border (in m).

    Returns:
        A list of trimesh.Trimesh objects that represent the border.
    """
    # compute thickness of the border
    thickness_x = (size[0] - inner_size[0]) / 2.0
    thickness_y = (size[1] - inner_size[1]) / 2.0
    # generate tri-meshes for the border
    # top/bottom border
    box_dims = (size[0], thickness_y, height)
    # -- top
    box_pos = (position[0], position[1] + inner_size[1] / 2.0 + thickness_y / 2.0, position[2])
    box_mesh_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # -- bottom
    box_pos = (position[0], position[1] - inner_size[1] / 2.0 - thickness_y / 2.0, position[2])
    box_mesh_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # left/right border
    box_dims = (thickness_x, inner_size[1], height)
    # -- left
    box_pos = (position[0] - inner_size[0] / 2.0 - thickness_x / 2.0, position[1], position[2])
    box_mesh_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # -- right
    box_pos = (position[0] + inner_size[0] / 2.0 + thickness_x / 2.0, position[1], position[2])
    box_mesh_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # return the tri-meshes
    return [box_mesh_left, box_mesh_right, box_mesh_top, box_mesh_bottom]


def make_box(
    length: float,
    width: float,
    height: float,
    center: tuple[float, float, float],
    max_yx_angle: float = 0,
    degrees: bool = True,
) -> trimesh.Trimesh:
    """Generate a box mesh with a random orientation.

    Args:
        length: The length (along x) of the box (in m).
        width: The width (along y) of the box (in m).
        height: The height of the cylinder (in m).
        center: The center of the cylinder (in m).
        max_yx_angle: The maximum angle along the y and x axis. Defaults to 0.
        degrees: Whether the angle is in degrees. Defaults to True.

    Returns:
        A trimesh.Trimesh object for the cylinder.
    """
    # create a pose for the cylinder
    transform = np.eye(4)
    transform[0:3, -1] = np.asarray(center)
    # -- create a random rotation
    euler_zyx = tf.Rotation.random().as_euler("zyx")  # returns rotation of shape (3,)
    # -- cap the rotation along the y and x axis
    if degrees:
        max_yx_angle = max_yx_angle / 180.0
    euler_zyx[1:] *= max_yx_angle
    # -- apply the rotation
    transform[0:3, 0:3] = tf.Rotation.from_euler("zyx", euler_zyx).as_matrix()
    # create the box
    dims = (length, width, height)
    return trimesh.creation.box(dims, transform=transform)


def make_cylinder(
    radius: float, height: float, center: tuple[float, float, float], max_yx_angle: float = 0, degrees: bool = True
) -> trimesh.Trimesh:
    """Generate a cylinder mesh with a random orientation.

    Args:
        radius: The radius of the cylinder (in m).
        height: The height of the cylinder (in m).
        center: The center of the cylinder (in m).
        max_yx_angle: The maximum angle along the y and x axis. Defaults to 0.
        degrees: Whether the angle is in degrees. Defaults to True.

    Returns:
        A trimesh.Trimesh object for the cylinder.
    """
    # create a pose for the cylinder
    transform = np.eye(4)
    transform[0:3, -1] = np.asarray(center)
    # -- create a random rotation
    euler_zyx = tf.Rotation.random().as_euler("zyx")  # returns rotation of shape (3,)
    # -- cap the rotation along the y and x axis
    if degrees:
        max_yx_angle = max_yx_angle / 180.0
    euler_zyx[1:] *= max_yx_angle
    # -- apply the rotation
    transform[0:3, 0:3] = tf.Rotation.from_euler("zyx", euler_zyx).as_matrix()
    # create the cylinder
    return trimesh.creation.cylinder(radius, height, sections=np.random.randint(4, 6), transform=transform)


def make_cone(
    radius: float, height: float, center: tuple[float, float, float], max_yx_angle: float = 0, degrees: bool = True
) -> trimesh.Trimesh:
    """Generate a cone mesh with a random orientation.

    Args:
        radius: The radius of the cone (in m).
        height: The height of the cone (in m).
        center: The center of the cone (in m).
        max_yx_angle: The maximum angle along the y and x axis. Defaults to 0.
        degrees: Whether the angle is in degrees. Defaults to True.

    Returns:
        A trimesh.Trimesh object for the cone.
    """
    # create a pose for the cylinder
    transform = np.eye(4)
    transform[0:3, -1] = np.asarray(center)
    # -- create a random rotation
    euler_zyx = tf.Rotation.random().as_euler("zyx")  # returns rotation of shape (3,)
    # -- cap the rotation along the y and x axis
    if degrees:
        max_yx_angle = max_yx_angle / 180.0
    euler_zyx[1:] *= max_yx_angle
    # -- apply the rotation
    transform[0:3, 0:3] = tf.Rotation.from_euler("zyx", euler_zyx).as_matrix()
    # create the cone
    return trimesh.creation.cone(radius, height, sections=np.random.randint(4, 6), transform=transform)
