# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.sim.spawners.spawner_cfg import SpawnerCfg
from omni.isaac.lab.utils import configclass

from . import sensors


@configclass
class PinholeCameraCfg(SpawnerCfg):
    """Configuration parameters for a USD camera prim with pinhole camera settings.

    For more information on the parameters, please refer to the `camera documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html>`__.

    .. note::
        The default values are taken from the `Replicator camera <https://docs.omniverse.nvidia.com/py/replicator/1.9.8/source/extensions/omni.replicator.core/docs/API.html#omni.replicator.core.create.camera>`__
        function.
    """

    func: Callable = sensors.spawn_camera

    projection_type: str = "pinhole"
    """Type of projection to use for the camera. Defaults to "pinhole".

    Note:
        Currently only "pinhole" is supported.
    """
    clipping_range: tuple[float, float] = (0.01, 1e6)
    """Near and far clipping distances (in m). Defaults to (0.01, 1e6).

    The minimum clipping range will shift the camera forward by the specified distance. Don't set it too high to
    avoid issues for distance related data types (e.g., ``distance_to_image_plane``).
    """
    focal_length: float = 24.0
    """Perspective focal length (in cm). Defaults to 24.0cm.

    Longer lens lengths narrower FOV, shorter lens lengths wider FOV.
    """
    focus_distance: float = 400.0
    """Distance from the camera to the focus plane (in m). Defaults to 400.0.

    The distance at which perfect sharpness is achieved.
    """
    f_stop: float = 0.0
    """Lens aperture. Defaults to 0.0, which turns off focusing.

    Controls Distance Blurring. Lower Numbers decrease focus range, larger numbers increase it.
    """
    horizontal_aperture: float = 20.955
    """Horizontal aperture (in mm). Defaults to 20.955mm.

    Emulates sensor/film width on a camera.

    Note:
        The default value is the horizontal aperture of a 35 mm spherical projector.
    """
    horizontal_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate horizontally. Defaults to 0.0."""
    vertical_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate vertically. Defaults to 0.0."""
    lock_camera: bool = True
    """Locks the camera in the Omniverse viewport. Defaults to True.

    If True, then the camera remains fixed at its configured transform. This is useful when wanting to view
    the camera output on the GUI and not accidentally moving the camera through the GUI interactions.
    """


@configclass
class FisheyeCameraCfg(PinholeCameraCfg):
    """Configuration parameters for a USD camera prim with `fish-eye camera`_ settings.

    For more information on the parameters, please refer to the
    `camera documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#fisheye-properties>`__.

    .. note::
        The default values are taken from the `Replicator camera <https://docs.omniverse.nvidia.com/py/replicator/1.9.8/source/extensions/omni.replicator.core/docs/API.html#omni.replicator.core.create.camera>`__
        function.

    .. _fish-eye camera: https://en.wikipedia.org/wiki/Fisheye_lens
    """

    func: Callable = sensors.spawn_camera

    projection_type: Literal[
        "fisheye_orthographic", "fisheye_equidistant", "fisheye_equisolid", "fisheye_polynomial", "fisheye_spherical"
    ] = "fisheye_polynomial"
    r"""Type of projection to use for the camera. Defaults to "fisheye_polynomial".

    Available options:

    - ``"fisheye_orthographic"``: Fisheye camera model using orthographic correction.
    - ``"fisheye_equidistant"``: Fisheye camera model using equidistant correction.
    - ``"fisheye_equisolid"``: Fisheye camera model using equisolid correction.
    - ``"fisheye_polynomial"``: Fisheye camera model with :math:`360^{\circ}` spherical projection.
    - ``"fisheye_spherical"``: Fisheye camera model with :math:`360^{\circ}` full-frame projection.
    """
    fisheye_nominal_width: float = 1936.0
    """Nominal width of fisheye lens model (in pixels). Defaults to 1936.0."""
    fisheye_nominal_height: float = 1216.0
    """Nominal height of fisheye lens model (in pixels). Defaults to 1216.0."""
    fisheye_optical_centre_x: float = 970.94244
    """Horizontal optical centre position of fisheye lens model (in pixels). Defaults to 970.94244."""
    fisheye_optical_centre_y: float = 600.37482
    """Vertical optical centre position of fisheye lens model (in pixels). Defaults to 600.37482."""
    fisheye_max_fov: float = 200.0
    """Maximum field of view of fisheye lens model (in degrees). Defaults to 200.0 degrees."""
    fisheye_polynomial_a: float = 0.0
    """First component of fisheye polynomial. Defaults to 0.0."""
    fisheye_polynomial_b: float = 0.00245
    """Second component of fisheye polynomial. Defaults to 0.00245."""
    fisheye_polynomial_c: float = 0.0
    """Third component of fisheye polynomial. Defaults to 0.0."""
    fisheye_polynomial_d: float = 0.0
    """Fourth component of fisheye polynomial. Defaults to 0.0."""
    fisheye_polynomial_e: float = 0.0
    """Fifth component of fisheye polynomial. Defaults to 0.0."""
    fisheye_polynomial_f: float = 0.0
    """Sixth component of fisheye polynomial. Defaults to 0.0."""


@configclass
class LidarCfg(SpawnerCfg):
    """
    Lidar Configuration Table

    +-------------+-----------+-----------------------------------+---------------------------+
    | Manufacturer| Model     | UI Name                           | Config Name               |
    +=============+===========+===================================+===========================+
    | HESAI       |PandarXT-32| PandarXT-32 10hz                  | Hesai_XT32_SD10           |
    +-------------+-----------+-----------------------------------+---------------------------+
    | Ouster      | OS0       | OS0 128 10hz @ 1024 resolution    | OS0_128ch10hz1024res      |
    |             |           | OS0 128 10hz @ 2048 resolution    | OS0_128ch10hz2048res      |
    |             |           | OS0 128 10hz @ 512 resolution     | OS0_128ch10hz512res       |
    |             |           | OS0 128 20hz @ 1024 resolution    | OS0_128ch20hz1024res      |
    |             |           | OS0 128 20hz @ 512 resolution     | OS0_128ch20hz512res       |
    +-------------+-----------+-----------------------------------+---------------------------+
    | Ouster      | OS1       | OS1 32 10hz @ 1024 resolution     | OS1_32ch10hz1024res       |
    |             |           | OS1 32 10hz @ 2048 resolution     | OS1_32ch10hz2048res       |
    |             |           | OS1 32 10hz @ 512 resolution      | OS1_32ch10hz512res        |
    |             |           | OS1 32 20hz @ 1024 resolution     | OS1_32ch20hz1024res       |
    |             |           | OS1 32 20hz @ 512 resolution      | OS1_32ch20hz512res        |
    +-------------+-----------+-----------------------------------+---------------------------+
    | SICK        | TiM781    | SICK TiM781                       | Sick_TiM781               |
    +-------------+-----------+-----------------------------------+---------------------------+
    | SLAMTEC     |RPLidar S2E| RPLidar S2E                       | RPLIDAR_S2E               |
    +-------------+-----------+-----------------------------------+---------------------------+
    | Velodyne    | VLS-128   | Velodyne VLS-128                  | Velodyne_VLS128           |
    +-------------+-----------+-----------------------------------+---------------------------+
    | ZVISION     | ML-30s+   | ML-30s+                           | ZVISION_ML30S             |
    |             | ML-Xs     | ML-Xs                             | ZVISION_MLXS              |
    +-------------+-----------+-----------------------------------+---------------------------+
    | NVIDIA      | Generic   | Rotating                          | Example_Rotary            |
    |             | Generic   | Solid State                       | Example_Solid_State       |
    |             | Debug     | Simple Solid State                | Simple_Example_Solid_State|
    +-------------+-----------+-----------------------------------+---------------------------+
    """
    func = sensors.spawn_lidar
    lidar_type: str = "Example_Rotary"
    class LidarType: 
        '''Class variables for autocompletion'''
        HESAI_PandarXT_32 = "Hesai_XT32_SD10"
        OUSTER_OS0_128_10HZ_1024RES = "OS0_128ch10hz1024res"
        OUSTER_OS0_128_10HZ_2048RES = "OS0_128ch10hz2048res"
        OUSTER_OS0_128_10HZ_512RES = "OS0_128ch10hz512res"
        OUSTER_OS0_128_20HZ_1024RES = "OS0_128ch20hz1024res"
        OUSTER_OS0_128_20HZ_512RES = "OS0_128ch20hz512res"
        OUSTER_OS1_32_10HZ_1024RES = "OS1_32ch10hz1024res"
        OUSTER_OS1_32_10HZ_2048RES = "OS1_32ch10hz2048res"
        OUSTER_OS1_32_10HZ_512RES = "OS1_32ch10hz512res"
        OUSTER_OS1_32_20HZ_1024RES = "OS1_32ch20hz1024res"
        OUSTER_OS1_32_20HZ_512RES = "OS1_32ch20hz512res"
        SICK_TIM781 = "Sick_TiM781"
        SLAMTEC_RPLIDAR_S2E = "RPLIDAR_S2E"
        VELODYNE_VLS128 = "Velodyne_VLS128"
        ZVISION_ML30S = "ZVISION_ML30S"
        ZVISION_MLXS = "ZVISION_MLXS"
        EXAMPLE_ROTARY = "Example_Rotary"
        EXAMPLE_SOLID_STATE = "Example_Solid_State"
        SIMPLE_EXAMPLE_SOLID_STATE = "Simple_Example_Solid_State"

