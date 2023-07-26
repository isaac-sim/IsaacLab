# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for sensors.

This class defines an interface for sensors similar to how the :class:`omni.isaac.orbit.robot.robot_base.RobotBase` class works.
Each sensor class should inherit from this class and implement the abstract methods.
"""


import torch
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext

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
        # Store the config
        self.cfg = cfg
        # Current timestamp (in seconds)
        self._timestamp: torch.Tensor
        # Timestamp from last update
        self._timestamp_last_update: torch.Tensor
        # ids of envs with outdated data
        self._is_outdated: torch.Tensor
        # Flag for whether the sensor is initialized
        self._is_initialized: bool = False
        # data object
        self._data: object = None

    """
    Properties
    """

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
    def timestamp(self) -> torch.Tensor:
        """Simulation time of the measurement (in seconds)."""
        return self._timestamp

    @property
    def data(self) -> Any:
        """Gets the data from the simulated sensor after updating it if necessary."""
        # update sensors if needed
        outdated_env_ids = self._is_outdated.nonzero().squeeze(-1)
        if len(outdated_env_ids) > 0:
            # obtain new data
            self._update_buffers(outdated_env_ids)
            # update the timestamp from last update
            self._timestamp_last_update[outdated_env_ids] = self._timestamp[outdated_env_ids]
            # set outdated flag to false for the updated sensors
            # TODO (from mayank): Why all of them are false? It is unclear. Shouldn't just be the ones that were updated?
            self._is_outdated[:] = False
        return self._data

    """
    Operations
    """

    def spawn(self, prim_path: str, *args, **kwargs):
        """Spawns the sensor into the stage.

        Args:
            prim_path (str): The path of the prim to attach the sensor to.
        """
        pass

    def initialize(self, env_prim_path: str = None):
        """Initializes the sensor handles and internal buffers.

        Args:
            env_prim_path (str, optional): The (templated) prim path expression to the environments where the sensors are created. Defaults to None.

        Raises:
            RuntimeError: If the simulation context is not initialized.
            RuntimeError: If no prims are found for the given prim path expression.
        """
        # Obtain Simulation Context
        sim = SimulationContext.instance()
        if sim is not None:
            self._device = sim.device
            self._sim_physics_dt = sim.get_physics_dt()
        else:
            raise RuntimeError("Simulation Context is not initialized!")
        # Count number of environments
        self._num_envs = len(prim_utils.find_matching_prim_paths(env_prim_path))
        # Boolean tensor indicating whether the sensor data has to be refreshed
        self._is_outdated = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        # Current timestamp (in seconds)
        self._timestamp = torch.zeros(self._num_envs, device=self._device)
        # Timestamp from last update
        self._timestamp_last_update = torch.zeros_like(self._timestamp)
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
        # Set all reset sensors to outdated so that they are updated when data is called the next time.
        self._is_outdated[env_ids] = True

    def update(self, dt: float, force_recompute: bool = False):
        # Update the timestamp for the sensors
        self._timestamp += dt
        self._is_outdated |= self._timestamp - self._timestamp_last_update + 1e-6 >= self.cfg.update_period
        # Update the buffers
        # TODO (from @mayank): Why is there a history length here when it doesn't mean anything in the sensor base?!?
        #   It is only for the contact sensor but there we should redefine the update function IMO.
        if force_recompute or (self.cfg.history_length > 0):
            # TODO (from @mayank): Why are we calling an attribute like this!? We should clean this up
            #   and make a function.
            self.data

    def debug_vis(self):
        """Visualizes the sensor data.

        This is an empty function that can be overridden by the derived class to visualize the sensor data.

        Note:
            Visualization of sensor data may add overhead to the simulation. It is recommended to disable
            visualization when running the simulation in headless mode.
        """
        pass

    """
    Implementation specific.
    """

    @abstractmethod
    def _update_buffers(self, env_ids: Optional[Sequence[int]] = None):
        """Fills the buffers of the sensor data.

        This function does not perform any time-based checks and directly fills the data into the data container.

        Args:
            env_ids (Optional[Sequence[int]]): The indices of the sensors that are ready to capture.
                If None, then all sensors are ready to capture. Defaults to None.
        """
        raise NotImplementedError
