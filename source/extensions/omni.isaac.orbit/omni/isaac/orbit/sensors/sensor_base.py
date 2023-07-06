# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for sensors.

This class defines an interface for sensors similar to how the :class:`omni.isaac.orbit.robot.robot_base.RobotBase` class works.
Each sensor class should inherit from this class and implement the abstract methods.
"""


from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence

from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.simulation_context import SimulationContext

from omni.isaac.orbit.utils.array import TensorData

from .sensor_base_cfg import SensorBaseCfg


class SensorBase(ABC):
    """The base class for implementing a sensor."""

    def __init__(self, cfg: SensorBaseCfg):
        """Initialize the sensor class.

        The sensor tick is the time between two sensor buffers. If the sensor tick is zero, then the sensor
        buffers are filled at every simulation step.

        Args:
            cfg (SensorBaseCfg): The configuration parameters for the sensor.
        """
        # Store the inputs
        self.cfg = cfg
        # Resolve the sensor update period
        self._update_period = 0.0 if cfg.update_freq == 0.0 else 1.0 / cfg.update_freq

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
        """The underlying view of the sensor."""
        return self._view

    @property
    def prim_paths(self) -> List[str]:
        """The paths to the sensor prims."""
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
    def frame(self) -> TensorData:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def timestamp(self) -> TensorData:
        """Simulation time of the measurement (in seconds)."""
        return self._timestamp

    @property
    @abstractmethod
    def data(self) -> Any:
        """The data from the simulated sensor."""
        raise NotImplementedError("The data property is not implemented!")

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

        Raises:
            RuntimeError: If the simulation context is not initialized.
            RuntimeError: If no prims are found for the given prim path expression.
        """
        # Obtain Simulation Context
        sim = SimulationContext.instance()
        if sim is not None:
            self._backend = sim.backend
            self._device = sim.device
            self._backend_utils = sim.backend_utils
            self._sim_physics_dt = sim.get_physics_dt()
        else:
            raise RuntimeError("Simulation Context is not initialized!")
        # Check that view is not None
        if self._view is None:
            self._view = XFormPrimView(prim_paths_expr, "sensor_view", reset_xform_properties=False)
        # Check is prims are found under the given prim path expression
        if self._view.count == 0:
            raise RuntimeError(f"No prims found for the given prim path expression: {prim_paths_expr}")
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

    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        """Resets the sensor internals.

        Args:
            env_ids (Optional[Sequence[int]], optional): The sensor ids to reset. Defaults to None.
        """
        # Resolve sensor ids
        if env_ids is None:
            env_ids = ...
        # Reset the timestamp for the sensors
        self._timestamp[env_ids] = 0.0
        self._timestamp_last_update[env_ids] = 0.0
        # Reset the frame count
        self._frame[env_ids] = 0

    def update_buffers(self, dt: float):
        """Updates the buffers at sensor frequency.

        This function performs time-based checks and fills the data into the data container. It
        calls the function :meth:`buffer()` to fill the data. The function :meth:`buffer()` should
        not be called directly.

        Args:
            dt (float): The simulation time-step.
        """
        # Update the timestamp for the sensors
        self._timestamp += dt
        # Check if the sensor is ready to capture
        env_ids = self._timestamp - self._timestamp_last_update >= self._update_period
        # Get the indices of the sensors that are ready to capture
        if self._backend == "torch":
            env_ids = env_ids.nonzero(as_tuple=False).squeeze(-1)
        elif self._backend == "numpy":
            env_ids = env_ids.nonzero()[0]
        else:
            raise NotImplementedError(f"Backend '{self._backend}' is not supported.")
        # Check if any sensor is ready to capture
        if len(env_ids) > 0:
            # Buffer the data
            self._buffer(env_ids)
            # Update the frame count
            self._frame[env_ids] += 1
            # Update capture time
            self._timestamp_last_update[env_ids] = self._timestamp[env_ids]

    def debug_vis(self):
        """Visualizes the sensor data.

        This is an empty function that can be overridden by the derived class to visualize the sensor data.

        Note:
            Visualization of sensor data may add overhead to the simulation. It is recommended to disable
            visualization when running the simulation in headless mode.

        Args:
            visualize (bool, optional): Whether to visualize the sensor data. Defaults to True.
        """
        pass

    """
    Implementation specific.
    """

    @abstractmethod
    def _buffer(self, env_ids: Optional[Sequence[int]] = None):
        """Fills the buffers of the sensor data.

        This function does not perform any time-based checks and directly fills the data into the data container.

        Args:
            env_ids (Optional[Sequence[int]]): The indices of the sensors that are ready to capture.
                If None, then all sensors are ready to capture. Defaults to None.
        """
        raise NotImplementedError
