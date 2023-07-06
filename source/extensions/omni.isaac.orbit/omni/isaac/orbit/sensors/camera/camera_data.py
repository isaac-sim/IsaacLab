import numpy as np
from dataclasses import dataclass
from tensordict import TensorDict
from typing import Any, Dict, List, Tuple, Union

from omni.isaac.orbit.utils.array import TensorData


@dataclass
class CameraData:
    """Data container for the camera sensor."""

    position: TensorData = None
    """Position of the sensor origin in world frame, following ROS convention.

    Shape is (N, 3) where ``N`` is the number of sensors.
    """
    orientation: TensorData = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in world frame, following ROS convention.

    Shape: (N, 4) where ``N`` is the number of sensors.
    """
    intrinsic_matrices: TensorData = None
    """The intrinsic matrices for the camera.

    Shape is (N, 3, 3) where ``N`` is the number of sensors.
    """
    image_shape: Tuple[int, int] = None
    """A tuple containing (height, width) of the camera sensor."""
    output: Union[Dict[str, np.ndarray], TensorDict] = None
    """The retrieved sensor data with sensor types as key.

    The format of the data is available in the `Replicator Documentation`_. For semantic-based data,
    this corresponds to the ``"data"`` key in the output of the sensor.

    .. _Replicator Documentation: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html#annotator-output
    """
    info: List[Dict[str, Any]] = None
    """The retrieved sensor info with sensor types as key.

    This contains extra information provided by the sensor such as semantic segmentation label mapping, prim paths.
    For semantic-based data, this corresponds to the ``"info"`` key in the output of the sensor. For other sensor
    types, the info is empty.
    """
