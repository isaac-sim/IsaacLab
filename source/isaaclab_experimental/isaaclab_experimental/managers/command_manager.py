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
import warp as wp
from prettytable import PrettyTable

from isaaclab_experimental.utils.warp.kernels import compute_reset_scale, count_masked

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import CommandTermCfg

# import omni.kit.app


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _sum_and_zero_masked(
    mask: wp.array(dtype=wp.bool),
    scale: wp.array(dtype=wp.float32),
    metric: wp.array(dtype=wp.float32),
    out_mean: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    if mask[env_id]:
        wp.atomic_add(out_mean, 0, metric[env_id] * scale[0])
        metric[env_id] = 0.0


@wp.kernel
def _zero_counter_masked(mask: wp.array(dtype=wp.bool), counter: wp.array(dtype=wp.int32)):
    env_id = wp.tid()
    if mask[env_id]:
        counter[env_id] = 0


@wp.kernel
def _step_time_left_and_build_resample_mask(
    time_left: wp.array(dtype=wp.float32),
    dt: wp.float32,
    out_mask: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()
    t = time_left[env_id] - dt
    time_left[env_id] = t
    out_mask[env_id] = t <= wp.float32(0.0)


@wp.kernel
def _resample_time_left_and_increment_counter(
    mask: wp.array(dtype=wp.bool),
    time_left: wp.array(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    rng_state: wp.array(dtype=wp.uint32),
    lower: wp.float32,
    upper: wp.float32,
):
    env_id = wp.tid()
    if mask[env_id]:
        s = rng_state[env_id]
        time_left[env_id] = wp.randf(s, lower, upper)
        rng_state[env_id] = s
        counter[env_id] = counter[env_id] + 1


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
        # -- metrics that can be used for logging (metric_name -> wp.array(num_envs,))
        self.metrics = dict()
        # -- time left before resampling
        self.time_left_wp = wp.zeros((self.num_envs,), dtype=wp.float32, device=self.device)
        # -- counter for the number of times the command has been resampled within the current episode
        self.command_counter_wp = wp.zeros((self.num_envs,), dtype=wp.int32, device=self.device)

        # reset/compute scratch buffers (Warp)
        self._reset_count_wp = wp.zeros((1,), dtype=wp.int32, device=self.device)
        self._reset_scale_wp = wp.zeros((1,), dtype=wp.float32, device=self.device)
        self._resample_mask_wp = wp.zeros((self.num_envs,), dtype=wp.bool, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # pre-allocated reset logging extras (filled during reset)
        self._reset_metric_mean_wp: dict[str, wp.array] = {}
        self._reset_extras: dict[str, torch.Tensor] = {}

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
    def command(self) -> torch.Tensor | wp.array:
        """The command tensor. Shape is (num_envs, command_dim)."""
        raise NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command generator has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    @property
    def reset_extras(self) -> dict[str, torch.Tensor]:
        """Pre-allocated reset logging extras for this command term."""
        return self._reset_extras

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
            # only enable debug_vis if omniverse is available
            from isaaclab.sim.simulation_context import SimulationContext

            sim_context = SimulationContext.instance()
            if not sim_context.has_omniverse_visualizer():
                return False
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                import omni.kit.app

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

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        env_mask: wp.array | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reset the command generator and log metrics.

        This function resets the command counter and resamples the command. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The specific environment indices to reset.
                If None, all environments are considered.
            env_mask: Boolean Warp mask of shape (num_envs,) selecting reset environments.
                If provided, takes precedence over ``env_ids``.

        Returns:
            A dictionary containing the information to log under the "{name}" key.
        """
        # Mask-first path: captured callers must provide env_mask.
        if env_mask is None or not isinstance(env_mask, wp.array):
            if wp.get_device().is_capturing:
                raise RuntimeError(
                    "CommandTerm.reset requires env_mask(wp.array[bool]) during capture. "
                    "Do not pass env_ids on captured paths."
                )
            env_mask = self._env.resolve_env_mask(env_ids=env_ids, env_mask=env_mask)

        # compute selected count and reset scale
        self._reset_count_wp.zero_()
        self._reset_scale_wp.zero_()
        wp.launch(kernel=count_masked, dim=self.num_envs, inputs=[env_mask, self._reset_count_wp], device=self.device)
        wp.launch(
            kernel=compute_reset_scale,
            dim=1,
            inputs=[self._reset_count_wp, 1.0, self._reset_scale_wp],
            device=self.device,
        )

        # update pre-allocated reset extras and clear selected metric rows
        for metric_name, metric_value_wp in self.metrics.items():
            out_mean_wp = self._reset_metric_mean_wp[metric_name]
            out_mean_wp.zero_()
            wp.launch(
                kernel=_sum_and_zero_masked,
                dim=self.num_envs,
                inputs=[env_mask, self._reset_scale_wp, metric_value_wp, out_mean_wp],
                device=self.device,
            )

        # set the command counter to zero
        wp.launch(
            kernel=_zero_counter_masked,
            dim=self.num_envs,
            inputs=[env_mask, self.command_counter_wp],
            device=self.device,
        )
        # resample the command
        self._resample(env_mask=env_mask)

        return self._reset_extras

    def _prepare_reset_extras(self):
        """Pre-allocate reset logging extras from metric definitions."""
        self._reset_metric_mean_wp = {}
        self._reset_extras = {}
        for metric_name, metric_value in self.metrics.items():
            if not isinstance(metric_value, wp.array):
                raise TypeError(
                    f"Metric '{metric_name}' must be a wp.array(dtype=wp.float32, shape=(num_envs,)). "
                    f"Received: {type(metric_value)}"
                )
            if metric_value.dtype != wp.float32 or metric_value.ndim != 1:
                raise TypeError(
                    f"Metric '{metric_name}' must be wp.float32 1D. "
                    f"Received dtype={metric_value.dtype}, ndim={metric_value.ndim}."
                )
            if metric_value.shape[0] != self.num_envs:
                raise ValueError(
                    f"Metric '{metric_name}' must have shape ({self.num_envs},), received {metric_value.shape}."
                )
            out_mean_wp = wp.zeros((1,), dtype=wp.float32, device=self.device)
            self._reset_metric_mean_wp[metric_name] = out_mean_wp
            self._reset_extras[metric_name] = wp.to_torch(out_mean_wp)[0]

    def compute(self, dt: float):
        """Compute the command.

        Args:
            dt: The time step passed since the last call to compute.
        """
        # update the metrics based on current state
        self._update_metrics()
        # reduce the time left before resampling and build resample mask
        wp.launch(
            kernel=_step_time_left_and_build_resample_mask,
            dim=self.num_envs,
            inputs=[self.time_left_wp, float(dt), self._resample_mask_wp],
            device=self.device,
        )
        # resample masked envs
        self._resample(env_mask=self._resample_mask_wp)
        # update the command
        self._update_command()

    """
    Helper functions.
    """

    def _resample(self, env_mask: wp.array):
        """Resample the command.

        This function resamples the command and time for which the command is applied for the
        specified environment mask.

        Args:
            env_mask: The boolean environment mask to resample.
        """
        if not isinstance(env_mask, wp.array):
            raise TypeError(f"env_mask must be a wp.array (got {type(env_mask)}).")
        if env_mask.dtype != wp.bool or env_mask.ndim != 1:
            raise TypeError(f"env_mask must be wp.bool 1D (got dtype={env_mask.dtype}, ndim={env_mask.ndim}).")
        if self._env.rng_state_wp is None:
            raise RuntimeError("Environment rng_state_wp is not initialized.")

        # resample time-left and increment command-counter for masked envs
        wp.launch(
            kernel=_resample_time_left_and_increment_counter,
            dim=self.num_envs,
            inputs=[
                env_mask,
                self.time_left_wp,
                self.command_counter_wp,
                self._env.rng_state_wp,
                float(self.cfg.resampling_time_range[0]),
                float(self.cfg.resampling_time_range[1]),
            ],
            device=self.device,
        )
        # resample command values for masked envs
        self._resample_command(env_mask)

    """
    Implementation specific functions.
    """

    @abstractmethod
    def _update_metrics(self):
        """Update the metrics based on the current state."""
        raise NotImplementedError

    @abstractmethod
    def _resample_command(self, env_mask: wp.array):
        """Resample the command for the specified masked environments."""
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

        # reset logging extras (persistent holder for orchestrator aggregation)
        self._reset_extras: dict[str, torch.Tensor] = {}
        for term_name, term in self._terms.items():
            for metric_name, metric_value in term.reset_extras.items():
                self._reset_extras[f"Metrics/{term_name}/{metric_name}"] = metric_value

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

    @property
    def reset_extras(self) -> dict[str, torch.Tensor]:
        """Persistent reset logging extras for command terms."""
        return self._reset_extras

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
        for name, term in self._terms.items():
            command = term.command
            if isinstance(command, wp.array):
                command = wp.to_torch(command)
            terms.append((name, command[env_idx].cpu().tolist()))
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

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        env_mask: wp.array | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reset the command terms and log their metrics.

        This function resets the command counter and resamples the command for each term. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The specific environment indices to reset.
                If None, all environments are considered.
            env_mask: Boolean Warp mask of shape (num_envs,) selecting reset environments.
                If provided, takes precedence over ``env_ids``.

        Returns:
            A dictionary containing the information to log under the "Metrics/{term_name}/{metric_name}" key.
        """
        # Mask-first path: captured callers must provide env_mask.
        if env_mask is None or not isinstance(env_mask, wp.array):
            if wp.get_device().is_capturing:
                raise RuntimeError(
                    "CommandManager.reset requires env_mask(wp.array[bool]) during capture. "
                    "Do not pass env_ids on captured paths."
                )
            env_mask = self._env.resolve_env_mask(env_ids=env_ids, env_mask=env_mask)

        for term in self._terms.values():
            # reset the command term
            term.reset(env_mask=env_mask)

        return self._reset_extras

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
        command = self._terms[name].command
        if isinstance(command, wp.array):
            return wp.to_torch(command)
        return command

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
            # pre-build reset extras once for capture-friendly reset logging
            term._prepare_reset_extras()
            # add class to dict
            self._terms[term_name] = term
