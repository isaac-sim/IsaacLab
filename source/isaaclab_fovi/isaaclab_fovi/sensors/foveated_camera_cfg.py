# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the FoveatedCamera sensor."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fovi.sensing.coords import RADIUS_NORMS, WARPED_CARTESIAN_STYLES
from fovi.sensing.projection import CameraCalibration

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .foveated_camera import FoveatedCamera


POOL_MODES = ("avg", "gaussian", "max", "sum")
"""fovi pooling modes for the outputs other than segmentation."""


@configclass
class FoveatedCameraCfg(CameraCfg):
    """Configuration for a foveated camera, whose retina samples densely where it looks and sparsely in the periphery.

    See :ref:`concepts_sensors_foveated_camera` for how each setting shapes the retina.
    """

    class_type: type[FoveatedCamera] | str = "{DIR}.foveated_camera:FoveatedCamera"

    # -- Foveation parameters ------------------------------------------------

    cmf_a_deg: float = 0.591
    """CMF constant *a* [deg]: smaller foveates more strongly; the default is an estimate from human vision."""

    camera_model: CameraCalibration | None = None
    """The physical camera as ``CameraModel`` fields; None uses the camera that :attr:`spawn` creates."""

    foveated_res: int = 60
    """Grid side of the retina; isotropic retinas use as many rings as keep at most its square of samples."""

    foveated_style: str = "warped_cartesian_as_grid"
    """Sampling style: the default grid publishes an upright image; sample-set styles publish a list of samples."""

    fov_type: str = "circular"
    """Region of the visual field the retina covers: ``"circular"``, ``"square"`` or ``"wang"``."""

    radius_norm: float = 2.0
    """Norm of the warped-Cartesian sampling shells: 2.0 for circular shells, infinity for square ones."""

    isotropic_plotting_type: str = "v1like"
    """Cortical layout of an isotropic retina's samples: ``"v1like"``, ``"schwartz"`` or ``"warp"``."""

    image_rotation_deg: float = 0.0
    """Counter-clockwise roll [deg] of the rays about the optical axis that undoes a rolled mount."""

    mask_to_sensor_fov: bool = False
    """Zero the samples outside the physical camera's image at the current gaze."""

    retinal_window_fraction: float = 1.0
    """Diameter of the retinal window as a fraction of the reference image side."""

    retinal_window_reference_side: str = "short"
    """Image side defining the retinal window diameter: ``short`` or ``long``."""

    # -- Optional retinal pooling (photoreceptor -> RGC two-layer sampling) --

    pool_enabled: bool = False
    """Render a denser photoreceptor grid and pool it to the retina."""

    pool_res_mult: float = 2
    """How many times denser the photoreceptor grid is than the retina; an integer for grid styles."""

    pool_k: int | None = None
    """Photoreceptors each sample pools: a square window's area for grid styles, else nearest neighbours."""

    pool_mode: str = "avg"
    """Pooling of non-segmentation outputs: ``"avg"``, ``"gaussian"``, ``"max"``, or ``"sum"`` for sample sets."""

    pool_sample_cortex: bool | str = "geodesic"
    """Neighbour distance for sample sets: ``"geodesic"`` on the cortex, ``True`` cortical, ``False`` visual."""

    pool_gauss_sigma: float = 10.0
    """Gaussian sigma for ``pool_mode="gaussian"``."""

    pool_int_mode: str = "modal"
    """Pooling of segmentation outputs: ``"modal"``, each sample's most common label among its neighbours."""

    # -- RTX-specific parameters (ignored by Newton backend) ----------------

    exit_texture_size: int | None = None
    """Side of the RTX inverse lens texture; None uses ``max(128, 4 * grid side)``."""

    # The camera sets height and width on its own copy of the configuration, from the retina's samples, before the
    # parent allocates buffers. The defaults only satisfy CameraCfg's required fields.
    width: int = 1
    height: int = 1

    update_latest_camera_pose: bool = True
    """Update the camera pose every frame. Newton and OVRTX require it: their pose update applies the gaze."""

    def __post_init__(self):
        super().__post_init__()
        # Reject incompatible geometry before the renderer creates the retina.
        if self.radius_norm not in RADIUS_NORMS:
            raise ValueError(f"radius_norm must be one of {RADIUS_NORMS}, got {self.radius_norm!r}.")
        if self.radius_norm != 2.0 and self.foveated_style not in WARPED_CARTESIAN_STYLES:
            raise ValueError("Non-default radius_norm requires a warped_cartesian style.")
        if self.radius_norm == math.inf and self.fov_type == "wang":
            raise ValueError("radius_norm=inf does not support fov_type='wang'.")
        if self.pool_mode not in POOL_MODES:
            raise ValueError(
                f"pool_mode must be one of {POOL_MODES}, got {self.pool_mode!r}; segmentation pools with pool_int_mode."
            )
        if self.pool_int_mode not in ("modal",):
            raise ValueError(
                f"pool_int_mode must be one of ('modal',), got {self.pool_int_mode!r}: segmentation"
                " pools by modal label, since other modes would mix labels into ones no object has."
            )
