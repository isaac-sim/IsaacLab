from __future__ import annotations

import torch
from dataclasses import dataclass
from tensordict import TensorDict

from .utils import convert_orientation_convention

@dataclass
class LidarData:
    """Data container for the lidar sensor."""

    ##
    # Frame state.
    ##

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame, following ROS convention.

    Shape is (N, 3) where N is the number of sensors.
    """

    quat_w_world: torch.Tensor = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in world frame, following the world coordinate frame

    .. note::
        World frame convention follows the camera aligned with forward axis +X and up axis +Z.

    Shape is (N, 4) where N is the number of sensors.
    """

    ##
    # LiDAR data
    ##

    output: TensorDict = None
    """The retrieved sensor data with data types as key. The available keys are:
    
    - ``"azimuth"``: The azimuth angle in radians for each column.
    - ``"depth"``: The distance from the sensor to the hit for each beam in uint16 and scaled by min and max distance.
    - ``"intensity"``: The observed specular intensity of each beam, 255 if hit, 0 if not.
    - ``"linear_depth"``: The distance from the sensor to the hit for each beam in meters.
    - ``"num_cols"``: The number of vertical scans of the sensor, 0 if error occurred.
    - ``"num_cols_ticked"``: The number of vertical scans the sensor completed in the last simulation step, 0 if error occurred. Generally only useful for lidars with a non-zero rotation speed.
    - ``"num_rows"``: The number of horizontal scans of the sensor, 0 if error occurred.
    - ``"point_cloud"``: The hit position in xyz relative to the sensor origin, not accounting for individual ray offsets.
    - ``"zenith"``: The zenith angle in radians for each row.
    """

    ##
    # Additional Frame orientation conventions
    ##

    @property
    def quat_w_ros(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following ROS convention.

        .. note::
            ROS convention follows the sensor aligned with forward axis +Z and up axis -Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        return convert_orientation_convention(self.quat_w_world, origin="world", target="ros")

    @property
    def quat_w_opengl(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following
        OpenGL / USD Camera convention.

        .. note::
            OpenGL convention follows the sensor aligned with forward axis -Z and up axis +Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        return convert_orientation_convention(self.quat_w_world, origin="world", target="opengl")
