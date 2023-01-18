# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for sensors in Omniverse workflows.

This class defines an interface similar to how the RobotBase class works.
"""


from abc import abstractmethod
from typing import Any


class SensorBase:
    """
    The base class for implementation of a sensor.

    Note:
        These sensors are not vectorized yet.

    Attributes:
        frame (int) - Frame number when the measurement took place.
        timestamp (float) - Simulation time of the measurement (in seconds).
        sensor_tick (float) - Simulation seconds between sensor buffers (ticks).
    """

    def __init__(self, sensor_tick: float = 0.0):
        """Initialize the sensor class.

        Args:
            sensor_tick (float, optional): Simulation seconds between sensor buffers. Defaults to 0.0.
        """
        # Copy arguments to class
        self._sensor_tick: float = sensor_tick
        # Current timestamp of animation (in seconds)
        self._timestamp: float = 0.0
        # Timestamp from last update
        self._timestamp_last_update: float = 0.0
        # Frame number when the measurement is taken
        self._frame: int = 0

    """
    Properties
    """

    @property
    def frame(self) -> int:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def timestamp(self) -> float:
        """Simulation time of the measurement (in seconds)."""
        return self._timestamp

    @property
    def sensor_tick(self) -> float:
        """Simulation seconds between sensor buffers (ticks)."""
        return self._sensor_tick

    @property
    def data(self) -> Any:
        """The data from the simulated sensor."""
        return None

    """
    Helpers
    """

    def set_visibility(self, visible: bool):
        """Set visibility of the instance in the scene.

        Note:
            Sensors are mostly XForms which do not have any mesh associated to them. Thus,
            overriding this method is optional.

        Args:
            visible (bool) -- Whether to make instance visible or invisible.
        """
        pass

    """
    Operations
    """

    @abstractmethod
    def spawn(self, parent_prim_path: str):
        """Spawns the sensor into the stage.

        Args:
            parent_prim_path (str): The path of the parent prim to attach sensor to.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self):
        """Initializes the sensor handles and internal buffers."""
        raise NotImplementedError

    def reset(self):
        """Resets the sensor internals."""
        # Set current time
        self._timestamp = 0.0
        # Set zero captures
        self._frame = 0

    def update(self, dt: float, *args, **kwargs):
        """Updates the buffers at sensor frequency.

        Args:
            dt (float): The simulation time-step.
            args (tuple): Other positional arguments passed to function :meth:`buffer()`.
            kwargs (dict): Other keyword arguments passed to function :meth:`buffer()`.
        """
        # Get current time
        self._timestamp += dt
        # Buffer the sensor data.
        if (self._timestamp - self._timestamp_last_update) >= self._sensor_tick:
            # Buffer the data
            self.buffer(*args, **kwargs)
            # Update the frame count
            self._frame += 1
            # Update capture time
            self._timestamp_last_update = self._timestamp

    @abstractmethod
    def buffer(self, *args, **kwargs):
        """Fills the buffers of the sensor data.

        This function does not perform any time-based checks and directly fills the data into the data container.

        Warning:
            Although this method is public, `update(dt)` should be the preferred way of filling buffers.
        """
        raise NotImplementedError
