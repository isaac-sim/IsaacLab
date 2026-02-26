# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command manager for generating and updating commands."""

from __future__ import annotations

import inspect
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

from isaaclab.utils.version import has_kit

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import CommandTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

if has_kit():
    import omni.kit.app


class CommandTerm(ManagerTermBase):
    """The base class for implementing a command term.

    A command term is used to generate commands for goal-conditioned tasks. For example,
    in the case of a goal-conditioned navigation task, the command term can be used to
    generate a target position for the robot to navigate to.

    It implements a resampling mechanism that allows the command to be resampled at a fixed
    frequency. The resampling frequency can be specified in the configuration object.
    Additionally, it is possible to assign a visualization function to the command term
    that can be used to visualize the command in the simulator.
    """

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # create buffers to store the command
        # -- metrics that can be used for logging
        self.metrics = dict()
        # -- time left before resampling
        self.time_left = torch.zeros(self.num_envs, device=self.device)
        # -- counter for the number of times the command has been resampled within the current episode
        self.command_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Properties
    """

    @property
    @abstractmethod
    def command(self) -> torch.Tensor:
        """The command tensor. Shape is (num_envs, command_dim)."""
        raise NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command generator has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the command data.

        Args:
            debug_vis: Whether to visualize the command data.

        Returns:
            Whether the debug visualization was successfully set. False if the command
            generator does not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command generator and log metrics.

        This function resets the command counter and resamples the command. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "{name}" key.
        """
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)

        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0

        # set the command counter to zero
        self.command_counter[env_ids] = 0
        # resample the command
        self._resample(env_ids)

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
        if len(env_ids) != 0:
            # resample the time left before resampling
            self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
            # resample the command
            self._resample_command(env_ids)
            # increment the command counter
            self.command_counter[env_ids] += 1

    """
    Implementation specific functions.
    """

    @abstractmethod
    def _update_metrics(self):
        """Update the metrics based on the current state."""
        raise NotImplementedError

    @abstractmethod
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the specified environments."""
        raise NotImplementedError

    @abstractmethod
    def _update_command(self):
        """Update the command based on the current state."""
        raise NotImplementedError

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        This function calls the visualization objects and sets the data to visualize into them.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")


class CommandManager(ManagerBase):
    """Manager for generating commands.

    The command manager is used to generate commands for an agent to execute. It makes it convenient to switch
    between different command generation strategies within the same environment. For instance, in an environment
    consisting of a quadrupedal robot, the command to it could be a velocity command or position command.
    By keeping the command generation logic separate from the environment, it is easy to switch between different
    command generation strategies.

    The command terms are implemented as classes that inherit from the :class:`CommandTerm` class.
    Each command generator term should also have a corresponding configuration class that inherits from the
    :class:`CommandTermCfg` class.
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the command manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, CommandTermCfg]``).
            env: The environment instance.
        """
        # create buffers to parse and store terms
        self._terms: dict[str, CommandTerm] = dict()

        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)
        # store the commands
        self._commands = dict()
        if self.cfg:
            self.cfg.debug_vis = False
            for term in self._terms.values():
                self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for the command manager."""
        msg = f"<CommandManager> contains {len(self._terms.values())} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Command Terms"
        table.field_names = ["Index", "Name", "Type"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.__class__.__name__])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active command terms."""
        return list(self._terms.keys())

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    """
    Operations.
    """

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """

        terms = []
        idx = 0
        for name, term in self._terms.items():
            terms.append((name, term.command[env_idx].cpu().tolist()))
            idx += term.command.shape[1]
        return terms

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the command data.

        Args:
            debug_vis: Whether to visualize the command data.

        Returns:
            Whether the debug visualization was successfully set. False if the command
            generator does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Reset the command terms and log their metrics.

        This function resets the command counter and resamples the command for each term. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "Metrics/{term_name}/{metric_name}" key.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for name, term in self._terms.items():
            # reset the command term
            metrics = term.reset(env_ids=env_ids)
            # compute the mean metric value
            for metric_name, metric_value in metrics.items():
                extras[f"Metrics/{name}/{metric_name}"] = metric_value
        # return logged information
        return extras

    def compute(self, dt: float):
        """Updates the commands.

        This function calls each command term managed by the class.

        Args:
            dt: The time-step interval of the environment.

        """
        # iterate over all the command terms
        for term in self._terms.values():
            # compute term's value
            term.compute(dt)

    def get_command(self, name: str) -> torch.Tensor:
        """Returns the command for the specified command term.

        Args:
            name: The name of the command term.

        Returns:
            The command tensor of the specified command term.
        """
        return self._terms[name].command

    def get_term(self, name: str) -> CommandTerm:
        """Returns the command term with the specified name.

        Args:
            name: The name of the command term.

        Returns:
            The command term with the specified name.
        """
        return self._terms[name]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, CommandTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type CommandTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, CommandTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type CommandType.")
            # add class to dict
            self._terms[term_name] = term
