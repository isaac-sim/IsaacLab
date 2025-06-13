# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import isaaclab.utils.sensors as sensor_utils
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

from . import sensors


@configclass
class PinholeCameraCfg(SpawnerCfg):
    """Configuration parameters for a USD camera prim with pinhole camera settings.

    For more information on the parameters, please refer to the `camera documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html>`__.

    ..note ::
        Focal length as well as the aperture sizes and offsets are set as a tenth of the world unit. In our case, the
        world unit is Meter s.t. all of these values are set in cm.

    .. note::
        The default values are taken from the `Replicator camera <https://docs.omniverse.nvidia.com/py/replicator/1.9.8/source/omni.replicator.core/docs/API.html#omni.replicator.core.create.camera>`__
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
    """Horizontal aperture (in cm). Defaults to 20.955 cm.

    Emulates sensor/film width on a camera.

    Note:
        The default value is the horizontal aperture of a 35 mm spherical projector.
    """

    vertical_aperture: float | None = None
    r"""Vertical aperture (in mm). Defaults to None.

    Emulates sensor/film height on a camera. If None, then the vertical aperture is calculated based on the
    horizontal aperture and the aspect ratio of the image to maintain squared pixels. This is calculated as:

    .. math::
        \text{vertical aperture} = \text{horizontal aperture} \times \frac{\text{height}}{\text{width}}
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

    @classmethod
    def from_intrinsic_matrix(
        cls,
        intrinsic_matrix: list[float],
        width: int,
        height: int,
        clipping_range: tuple[float, float] = (0.01, 1e6),
        focal_length: float | None = None,
        focus_distance: float = 400.0,
        f_stop: float = 0.0,
        projection_type: str = "pinhole",
        lock_camera: bool = True,
    ) -> PinholeCameraCfg:
        r"""Create a :class:`PinholeCameraCfg` class instance from an intrinsic matrix.

        The intrinsic matrix is a 3x3 matrix that defines the mapping between the 3D world coordinates and
        the 2D image. The matrix is defined as:

        .. math::
            I_{cam} = \begin{bmatrix}
            f_x & 0 & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
            \\end{bmatrix},

        where :math:`f_x` and :math:`f_y` are the focal length along x and y direction, while :math:`c_x` and :math:`c_y` are the
        principle point offsets along x and y direction respectively.

        Args:
            intrinsic_matrix: Intrinsic matrix of the camera in row-major format.
                The matrix is defined as [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]. Shape is (9,).
            width: Width of the image (in pixels).
            height: Height of the image (in pixels).
            clipping_range: Near and far clipping distances (in m). Defaults to (0.01, 1e6).
            focal_length: Perspective focal length (in cm) used to calculate pixel size. Defaults to None. If None
                focal_length will be calculated 1 / width.
            focus_distance: Distance from the camera to the focus plane (in m). Defaults to 400.0 m.
            f_stop: Lens aperture. Defaults to 0.0, which turns off focusing.
            projection_type: Type of projection to use for the camera. Defaults to "pinhole".
            lock_camera: Locks the camera in the Omniverse viewport. Defaults to True.

        Returns:
            An instance of the :class:`PinholeCameraCfg` class.
        """
        # raise not implemented error is projection type is not pinhole
        if projection_type != "pinhole":
            raise NotImplementedError("Only pinhole projection type is supported.")

        usd_camera_params = sensor_utils.convert_camera_intrinsics_to_usd(
            intrinsic_matrix=intrinsic_matrix, height=height, width=width, focal_length=focal_length
        )

        return cls(
            projection_type=projection_type,
            clipping_range=clipping_range,
            focal_length=usd_camera_params["focal_length"],
            focus_distance=focus_distance,
            f_stop=f_stop,
            horizontal_aperture=usd_camera_params["horizontal_aperture"],
            vertical_aperture=usd_camera_params["vertical_aperture"],
            horizontal_aperture_offset=usd_camera_params["horizontal_aperture_offset"],
            vertical_aperture_offset=usd_camera_params["vertical_aperture_offset"],
            lock_camera=lock_camera,
        )


@configclass
class FisheyeCameraCfg(PinholeCameraCfg):
    """Configuration parameters for a USD camera prim with `fish-eye camera`_ settings.

    For more information on the parameters, please refer to the
    `camera documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#fisheye-properties>`__.

    .. note::
        The default values are taken from the `Replicator camera <https://docs.omniverse.nvidia.com/py/replicator/1.9.8/source/omni.replicator.core/docs/API.html#omni.replicator.core.create.camera>`__
        function.

    .. _fish-eye camera: https://en.wikipedia.org/wiki/Fisheye_lens
    """

    func: Callable = sensors.spawn_camera

    projection_type: Literal[
        "fisheyePolynomial",
        "fisheyeSpherical",
        "fisheyeKannalaBrandtK3",
        "fisheyeRadTanThinPrism",
        "omniDirectionalStereo",
    ] = "fisheyePolynomial"
    r"""Type of projection to use for the camera. Defaults to "fisheyePolynomial".

    Available options:

    - ``"fisheyePolynomial"``: Fisheye camera model with :math:`360^{\circ}` spherical projection.
    - ``"fisheyeSpherical"``: Fisheye camera model with :math:`360^{\circ}` full-frame projection.
    - ``"fisheyeKannalaBrandtK3"``: Fisheye camera model using the Kannala-Brandt K3 distortion model.
    - ``"fisheyeRadTanThinPrism"``: Fisheye camera model that combines radial and tangential distortions.
    - ``"omniDirectionalStereo"``: Fisheye camera model supporting :math:`360^{\circ}` stereoscopic imaging.
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
