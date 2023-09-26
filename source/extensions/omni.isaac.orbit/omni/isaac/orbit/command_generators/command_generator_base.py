# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for command generators.

This class defines an interface for command generators that can be used for goal-conditioned
tasks. Each command generator class should inherit from this class and implement the abstract
methods.
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import omni.kit.app

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

    from .command_generator_cfg import CommandGeneratorBaseCfg


class CommandGeneratorBase(ABC):
    """The base class for implementing a command generator.

    A command generator is used to generate commands for goal-conditioned tasks. For example,
    in the case of a goal-conditioned navigation task, the command generator can be used to
    generate a target position for the robot to navigate to.

    The command generator implements a resampling mechanism that allows the command to be
    resampled at a fixed frequency. The resampling frequency can be specified in the
    configuration object. Additionally, it is possible to assign a visualization function
    to the command generator that can be used to visualize the command in the simulator.
    """

    def __init__(self, cfg: CommandGeneratorBaseCfg, env: BaseEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # store the inputs
        self.cfg = cfg
        self._env = env
        # create buffers to store the command
        # -- metrics that can be used for logging
        self.metrics = dict()
        # -- time left before resampling
        self.time_left = torch.zeros(self.num_envs, device=self.device)
        # -- counter for the number of times the command has been resampled within the current episode
        self.command_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # add callback for debug visualization
        if self.cfg.debug_vis:
            app_interface = omni.kit.app.get_app_interface()
            self._debug_visualization_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                self._debug_vis_callback
            )
        else:
            self._debug_visualization_handle = None

    def __del__(self):
        """Unsubscribe from the callbacks."""
        if self._debug_visualization_handle is not None:
            self._debug_visualization_handle.unsubscribe()

    """
    Properties
    """

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._env.num_envs

    @property
    def device(self) -> str:
        """Device on which to perform computations."""
        return self._env.device

    @property
    @abstractmethod
    def command(self) -> torch.Tensor:
        """The command tensor. Shape is (num_envs, command_dim)."""
        raise NotImplementedError

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the command data.

        Args:
            debug_vis: Whether to visualize the command data.

        Raises:
            RuntimeError: If the command debug visualization is not enabled.
        """
        if not self.cfg.debug_vis:
            raise RuntimeError("Debug visualization is not enabled for this sensor.")

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command generator and log metrics.

        This function resets the command counter and resamples the command. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "Metrics/{name}" key.
        """
        # resolve the environment IDs
        if env_ids is None:
            env_ids = ...
        # set the command counter to zero
        self.command_counter[env_ids] = 0
        # resample the command
        self._resample(env_ids)
        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[f"Metrics/{metric_name}"] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0
        return extras

    def compute(self, dt: float):
        """Compute the command.

        Args:
            dt: The time step passed since the last call to compute.
        """
        # update the metrics based on current state
        self._update_metrics()
        # reduce the time left before resampling
        self.time_left -= dt
        # resample the command if necessary
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        # update the command
        self._update_command()

    """
    Helper functions.
    """

    def _resample(self, env_ids: Sequence[int]):
        """Resample the command.

        This function resamples the command and time for which the command is applied for the
        specified environment indices.

        Args:
            env_ids: The list of environment IDs to resample.
        """
        # resample the time left before resampling
        self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
        # increment the command counter
        self.command_counter[env_ids] += 1
        # resample the command
        self._resample_command(env_ids)

    """
    Simulation callbacks.
    """

    def _debug_vis_callback(self, event):
        """Visualizes the sensor data."""
        self._debug_vis_impl()

    """
    Implementation specific functions.
    """

    @abstractmethod
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the specified environments."""
        raise NotImplementedError

    @abstractmethod
    def _update_command(self):
        """Update the command based on the current state."""
        raise NotImplementedError

    @abstractmethod
    def _update_metrics(self):
        """Update the metrics based on the current state."""
        raise NotImplementedError

    def _debug_vis_impl(self):
        """Visualize the command in the simulator.

        This is an optional function that can be used to visualize the command in the simulator.
        """
        pass
