# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for sensors.

This class defines an interface for sensors similar to how the :class:`omni.isaac.orbit.robot.robot_base.RobotBase` class works.
Each sensor class should inherit from this class and implement the abstract methods.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any
from warnings import warn

import carb


class SensorBase:
    """The base class for implementing a sensor.

    Note:
        These sensors are not vectorized yet.
    """

    def __init__(self, sensor_tick: float = 0.0):
        """Initialize the sensor class.

        The sensor tick is the time between two sensor buffers. If the sensor tick is zero, then the sensor
        buffers are filled at every simulation step.

        Args:
            sensor_tick: Simulation seconds between sensor buffers. Defaults to 0.0.
        """
        # print warning to notify user that the sensor is not vectorized
        carb.log_warn("This implementation of the sensor is not vectorized yet. Please use the vectorized version.")
        # Copy arguments to class
        self._sensor_tick: float = sensor_tick
        # Current timestamp of animation (in seconds)
        self._timestamp: float = 0.0
        # Timestamp from last update
        self._timestamp_last_update: float = 0.0
        # Frame number when the measurement is taken
        self._frame: int = 0

    def __init_subclass__(cls, **kwargs):
        """This throws a deprecation warning on subclassing."""
        warn(f"{cls.__name__} will be deprecated from v1.0.", DeprecationWarning, stacklevel=1)
        super().__init_subclass__(**kwargs)

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
            visible: Whether to make instance visible or invisible.
        """
        pass

    """
    Operations
    """

    @abstractmethod
    def spawn(self, parent_prim_path: str):
        """Spawns the sensor into the stage.

        Args:
            parent_prim_path: The path of the parent prim to attach sensor to.
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
        self._timestamp_last_update = 0.0
        # Set zero captures
        self._frame = 0

    def update(self, dt: float, *args, **kwargs):
        """Updates the buffers at sensor frequency.

        This function performs time-based checks and fills the data into the data container. It
        calls the function :meth:`buffer()` to fill the data. The function :meth:`buffer()` should
        not be called directly.

        Args:
            dt: The simulation time-step.
            args: Other positional arguments passed to function :meth:`buffer()`.
            kwargs: Other keyword arguments passed to function :meth:`buffer()`.
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
