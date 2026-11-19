# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the FoveatedCamera sensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fovi.sensing.projection import CameraCalibration

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .foveated_camera import FoveatedCamera


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
    """Side length of the published warped-Cartesian grid."""

    image_rotation_deg: float = 0.0
    """Counter-clockwise roll [deg] of the rays about the optical axis that undoes a rolled mount."""

    mask_to_sensor_fov: bool = False
    """Zero the samples outside the physical camera's image at the current gaze."""

    retinal_window_fraction: float = 1.0
    """Diameter of the retinal window as a fraction of the reference image side."""

    retinal_window_reference_side: str = "short"
    """Image side defining the retinal window diameter: ``short`` or ``long``."""

    # -- RTX-specific parameters (ignored by Newton backend) ----------------

    exit_texture_size: int | None = None
    """Side of the RTX inverse lens texture; None uses ``max(128, 4 * grid side)``."""

    # The camera sets height and width on its own copy of the configuration, from the retina's samples, before the
    # parent allocates buffers. The defaults only satisfy CameraCfg's required fields.
    width: int = 1
    height: int = 1

    update_latest_camera_pose: bool = True
    """Update the camera pose every frame. Newton and OVRTX require it: their pose update applies the gaze."""
