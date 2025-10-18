from collections.abc import Sequence
import torch 
from abc import abstractmethod
import numpy as np 

class MujocoBaseSensor():
    def __init__(self, env_cfg):
        self._device = env_cfg.sim.device
        self._dt = env_cfg.sim.dt
        self.sensor_cfg = env_cfg
        
    @abstractmethod
    def _initialize_impl(self):
        self._num_envs = 1
        # Boolean tensor indicating whether the sensor data has to be refreshed
        self._is_outdated = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        # Current timestamp (in seconds)
        self._timestamp = torch.zeros(self._num_envs, device=self._device)
        # Timestamp from last update
        self._timestamp_last_update = torch.zeros_like(self._timestamp)

    def reset(self, env_ids: Sequence[int] | None = None):
        # Resolve sensor ids
        if env_ids is None:
            env_ids = slice(None)
        # Reset the timestamp for the sensors
        self._timestamp[env_ids] = 0.0
        self._timestamp_last_update[env_ids] = 0.0
        # Set all reset sensors to outdated so that they are updated when data is called the next time.
        self._is_outdated[env_ids] = True

    def update(self, dt: float, force_recompute: bool = True):
        # Update the timestamp for the sensors
        self._dt = dt
        self._timestamp += dt
        self._is_outdated |= self._timestamp - self._timestamp_last_update + 1e-6 >= self.sensor_cfg.update_period
        if force_recompute or (self.sensor_cfg.history_length > 0):
            self._update_outdated_buffers()

    @abstractmethod
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        raise NotImplementedError

    def _update_outdated_buffers(self):
        outdated_env_ids = self._is_outdated.nonzero().squeeze(-1)
        if len(outdated_env_ids) > 0:
            # obtain new data
            self._update_buffers_impl(outdated_env_ids)
            # update the timestamp from last update
            self._timestamp_last_update[outdated_env_ids] = self._timestamp[outdated_env_ids]
            # set outdated flag to false for the updated sensors
            self._is_outdated[outdated_env_ids] = False
