# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for sensors.

This class defines an interface for sensors similar to how the :class:`omni.isaac.orbit.robot.robot_base.RobotBase` class works.
Each sensor class should inherit from this class and implement the abstract methods.
"""


from abc import abstractmethod
from typing import Any, List, Optional, Sequence

from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.simulation_context import SimulationContext

from omni.isaac.orbit.utils import TensorData


class SensorBase:
    """The base class for implementing a sensor."""

    def __init__(self, sensor_tick: float = 0.0):
        """Initialize the sensor class.

        The sensor tick is the time between two sensor buffers. If the sensor tick is zero, then the sensor
        buffers are filled at every simulation step.

        Args:
            sensor_tick (float, optional): Simulation seconds between sensor buffers. Defaults to 0.0.
        """
        # Copy arguments to class
        self._sensor_tick: float = sensor_tick
        # Obtain Simulation Context
        sim = SimulationContext.instance()
        if sim is not None:
            self._backend = sim.backend
            self._device = sim.device
            self._backend_utils = sim.backend_utils
        else:
            import omni.isaac.core.utils.numpy as np_utils

            self._backend = "numpy"
            self._device = None
            self._backend_utils = np_utils
        # Sensor view -- This can be used to access the underlying XFormPrimView
        # Note: This is not initialized here. It is initialized in the derived class.
        self._view: XFormPrimView = None
        # Current timestamp of animation (in seconds)
        self._timestamp: TensorData
        # Timestamp from last update
        self._timestamp_last_update: TensorData
        # Frame number when the measurement is taken
        self._frame: TensorData
        # Flag for whether the sensor is initialized
        self._is_initialized: bool = False

    """
    Properties
    """

    @property
    def view(self) -> XFormPrimView:
        """The underlying XFormPrimView of the sensor."""
        return self._view

    @property
    def prim_paths(self) -> List[str]:
        """The path to the camera prim."""
        return self._view.prim_paths

    @property
    def count(self) -> int:
        """Number of prims encapsulated."""
        return self._view.count

    @property
    def device(self) -> str:
        """Device where the sensor is running."""
        return self._device

    @property
    def sensor_tick(self) -> float:
        """Simulation seconds between sensor buffers (ticks)."""
        return self._sensor_tick

    @property
    def frame(self) -> TensorData:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def timestamp(self) -> TensorData:
        """Simulation time of the measurement (in seconds)."""
        return self._timestamp

    @property
    def data(self) -> Any:
        """The data from the simulated sensor."""
        return None  # noqa: R501

    """
    Helpers
    """

    def set_visibility(self, visible: bool):
        """Set visibility of the instance in the scene.

        Note:
            Sensors are mostly XForms which do not have any mesh associated to them. Thus,
            overriding this method is optional.

        Args:
            visible (bool): Whether to make instance visible or invisible.
        """
        # check camera prim
        if len(self.count) == 0:
            raise RuntimeError("Camera prims are None. Please call 'initialize(...)' first.")
        # set visibility
        self.view.set_visibilities([visible] * self.count)

    """
    Operations
    """

    @abstractmethod
    def spawn(self, prim_path: str, *args, **kwargs):
        """Spawns the sensor into the stage.

        Args:
            prim_path (str): The path of the prim to attach the sensor to.
        """
        raise NotImplementedError

    def initialize(self, prim_paths_expr: str = None):
        """Initializes the sensor handles and internal buffers.

        Args:
            prim_paths_expr (str, optional): The prim path expression for the sensors. Defaults to None.
        """
        # Check that view is not None
        if self._view is None:
            self._view = XFormPrimView(prim_paths_expr, "sensor_view", reset_xform_properties=True)
        # Create constant for the number of sensors
        self._ALL_INDICES = self._backend_utils.resolve_indices(None, self.count, device=self.device)
        # Current timestamp of animation (in seconds)
        self._timestamp = self._backend_utils.create_zeros_tensor((self.count,), "float32", self.device)
        # Timestamp from last update
        self._timestamp_last_update = self._backend_utils.create_zeros_tensor((self.count,), "float32", self.device)
        # Frame number when the measurement is taken
        self._frame = self._backend_utils.create_zeros_tensor((self.count,), "int64", self.device)
        # Set the initialized flag
        self._is_initialized = True

    def reset_buffers(self, sensor_ids: Optional[Sequence[int]] = None):
        """Resets the sensor internals.

        Args:
            sensor_ids (Sequence[int], optional): The sensor ids to reset. Defaults to None.
        """
        # Resolve sensor ids
        if sensor_ids is None:
            sensor_ids = self._ALL_INDICES
        # Reset the timestamp for the sensors
        self._timestamp[sensor_ids] = 0.0
        self._timestamp_last_update[sensor_ids] = 0.0
        # Reset the frame count
        self._frame[sensor_ids] = 0

    def update_buffers(self, dt: float, *args, **kwargs):
        """Updates the buffers at sensor frequency.

        This function performs time-based checks and fills the data into the data container. It
        calls the function :meth:`buffer()` to fill the data. The function :meth:`buffer()` should
        not be called directly.

        Args:
            dt (float): The simulation time-step.
            args (tuple): Other positional arguments passed to function :meth:`buffer()`.
            kwargs (dict): Other keyword arguments passed to function :meth:`buffer()`.
        """
        # Update the timestamp for the sensors
        self._timestamp += dt
        # Check if the sensor is ready to capture
        sensor_ids = self._timestamp - self._timestamp_last_update >= self._sensor_tick
        # Get the indices of the sensors that are ready to capture
        if self._backend == "torch":
            sensor_ids = sensor_ids.nonzero(as_tuple=False).squeeze(-1)
        elif self._backend == "numpy":
            sensor_ids = sensor_ids.nonzero()[0]
        else:
            raise NotImplementedError(f"Backend '{self._backend}' is not supported.")
        # Check if any sensor is ready to capture
        if len(sensor_ids) > 0:
            # Buffer the data
            self.buffer(sensor_ids, *args, **kwargs)
            # Update the frame count
            self._frame[sensor_ids] += 1
            # Update capture time
            self._timestamp_last_update[sensor_ids] = self._timestamp[sensor_ids]

    @abstractmethod
    def buffer(self, sensor_ids: Optional[Sequence[int]] = None, *args, **kwargs):
        """Fills the buffers of the sensor data.

        This function does not perform any time-based checks and directly fills the data into the data container.

        Warning:
            Although this method is public, `update(dt)` should be the preferred way of filling buffers.

        Args:
            sensor_ids (Optional[Sequence[int]]): The indices of the sensors that are ready to capture.
                If None, then all sensors are ready to capture. Defaults to None.
        """
        raise NotImplementedError
