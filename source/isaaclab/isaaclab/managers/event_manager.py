# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event manager for orchestrating operations based on different simulation events."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.log

from .manager_base import ManagerBase
from .manager_term_cfg import EventTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class EventManager(ManagerBase):
    """Manager for orchestrating operations based on different simulation events.

    The event manager applies operations to the environment based on different simulation events. For example,
    changing the masses of objects or their friction coefficients during initialization/ reset, or applying random
    pushes to the robot at a fixed interval of steps. The user can specify several modes of events to fine-tune the
    behavior based on when to apply the event.

    The event terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each event term should instantiate the :class:`EventTermCfg` class.

    Event terms can be grouped by their mode. The mode is a user-defined string that specifies when
    the event term should be applied. This provides the user complete control over when event
    terms should be applied.

    For a typical training process, you may want to apply events in the following modes:

    - "prestartup": Event is applied once at the beginning of the training before the simulation starts.
      This is used to randomize USD-level properties of the simulation stage.
    - "startup": Event is applied once at the beginning of the training once simulation is started.
    - "reset": Event is applied at every reset.
    - "interval": Event is applied at pre-specified intervals of time.

    However, you can also define your own modes and use them in the training process as you see fit.
    For this you will need to add the triggering of that mode in the environment implementation as well.

    .. note::

        The triggering of operations corresponding to the mode ``"interval"`` are the only mode that are
        directly handled by the manager itself. The other modes are handled by the environment implementation.

    """

    _env: ManagerBasedEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the event manager.

        Args:
            cfg: A configuration object or dictionary (``dict[str, EventTermCfg]``).
            env: An environment object.
        """
        # create buffers to parse and store terms
        self._mode_term_names: dict[str, list[str]] = dict()
        self._mode_term_cfgs: dict[str, list[EventTermCfg]] = dict()
        self._mode_class_term_cfgs: dict[str, list[EventTermCfg]] = dict()

        # call the base class (this will parse the terms config)
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for event manager."""
        msg = f"<EventManager> contains {len(self._mode_term_names)} active terms.\n"

        # add info on each mode
        for mode in self._mode_term_names:
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Event Terms in Mode: '{mode}'"
            # add table headers based on mode
            if mode == "interval":
                table.field_names = ["Index", "Name", "Interval time range (s)"]
                table.align["Name"] = "l"
                for index, (name, cfg) in enumerate(zip(self._mode_term_names[mode], self._mode_term_cfgs[mode])):
                    table.add_row([index, name, cfg.interval_range_s])
            else:
                table.field_names = ["Index", "Name"]
                table.align["Name"] = "l"
                for index, name in enumerate(self._mode_term_names[mode]):
                    table.add_row([index, name])
            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active event terms.

        The keys are the modes of event and the values are the names of the event terms.
        """
        return self._mode_term_names

    @property
    def available_modes(self) -> list[str]:
        """Modes of events."""
        return list(self._mode_term_names.keys())

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # call all terms that are classes
        for mode_cfg in self._mode_class_term_cfgs.values():
            for term_cfg in mode_cfg:
                term_cfg.func.reset(env_ids=env_ids)

        # resolve number of environments
        if env_ids is None:
            num_envs = self._env.num_envs
        else:
            num_envs = len(env_ids)
        # if we are doing interval based events then we need to reset the time left
        # when the episode starts. otherwise the counter will start from the last time
        # for that environment
        if "interval" in self._mode_term_cfgs:
            for index, term_cfg in enumerate(self._mode_term_cfgs["interval"]):
                # sample a new interval and set that as time left
                # note: global time events are based on simulation time and not episode time
                #   so we do not reset them
                if not term_cfg.is_global_time:
                    lower, upper = term_cfg.interval_range_s
                    sampled_interval = torch.rand(num_envs, device=self.device) * (upper - lower) + lower
                    self._interval_term_time_left[index][env_ids] = sampled_interval

        # nothing to log here
        return {}

    def apply(
        self,
        mode: str,
        env_ids: Sequence[int] | None = None,
        dt: float | None = None,
        global_env_step_count: int | None = None,
    ):
        """Calls each event term in the specified mode.

        This function iterates over all the event terms in the specified mode and calls the function
        corresponding to the term. The function is called with the environment instance and the environment
        indices to apply the event to.

        For the "interval" mode, the function is called when the time interval has passed. This requires
        specifying the time step of the environment.

        For the "reset" mode, the function is called when the mode is "reset" and the total number of environment
        steps that have happened since the last trigger of the function is equal to its configured parameter for
        the number of environment steps between resets.

        Args:
            mode: The mode of event.
            env_ids: The indices of the environments to apply the event to.
                Defaults to None, in which case the event is applied to all environments when applicable.
            dt: The time step of the environment. This is only used for the "interval" mode.
                Defaults to None to simplify the call for other modes.
            global_env_step_count: The total number of environment steps that have happened. This is only used
                for the "reset" mode. Defaults to None to simplify the call for other modes.

        Raises:
            ValueError: If the mode is ``"interval"`` and the time step is not provided.
            ValueError: If the mode is ``"interval"`` and the environment indices are provided. This is an undefined
                behavior as the environment indices are computed based on the time left for each environment.
            ValueError: If the mode is ``"reset"`` and the total number of environment steps that have happened
                is not provided.
        """
        # check if mode is valid
        if mode not in self._mode_term_names:
            omni.log.warn(f"Event mode '{mode}' is not defined. Skipping event.")
            return

        # check if mode is interval and dt is not provided
        if mode == "interval" and dt is None:
            raise ValueError(f"Event mode '{mode}' requires the time-step of the environment.")
        if mode == "interval" and env_ids is not None:
            raise ValueError(
                f"Event mode '{mode}' does not require environment indices. This is an undefined behavior"
                " as the environment indices are computed based on the time left for each environment."
            )
        # check if mode is reset and env step count is not provided
        if mode == "reset" and global_env_step_count is None:
            raise ValueError(f"Event mode '{mode}' requires the total number of environment steps to be provided.")

        # iterate over all the event terms
        for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):
            if mode == "interval":
                # extract time left for this term
                time_left = self._interval_term_time_left[index]
                # update the time left for each environment
                time_left -= dt

                # check if the interval has passed and sample a new interval
                # note: we compare with a small value to handle floating point errors
                if term_cfg.is_global_time:
                    if time_left < 1e-6:
                        lower, upper = term_cfg.interval_range_s
                        sampled_interval = torch.rand(1) * (upper - lower) + lower
                        self._interval_term_time_left[index][:] = sampled_interval

                        # call the event term (with None for env_ids)
                        term_cfg.func(self._env, None, **term_cfg.params)
                else:
                    valid_env_ids = (time_left < 1e-6).nonzero().flatten()
                    if len(valid_env_ids) > 0:
                        lower, upper = term_cfg.interval_range_s
                        sampled_time = torch.rand(len(valid_env_ids), device=self.device) * (upper - lower) + lower
                        self._interval_term_time_left[index][valid_env_ids] = sampled_time

                        # call the event term
                        term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
            elif mode == "reset":
                # obtain the minimum step count between resets
                min_step_count = term_cfg.min_step_count_between_reset
                # resolve the environment indices
                if env_ids is None:
                    env_ids = slice(None)

                # We bypass the trigger mechanism if min_step_count is zero, i.e. apply term on every reset call.
                # This should avoid the overhead of checking the trigger condition.
                if min_step_count == 0:
                    self._reset_term_last_triggered_step_id[index][env_ids] = global_env_step_count
                    self._reset_term_last_triggered_once[index][env_ids] = True

                    # call the event term with the environment indices
                    term_cfg.func(self._env, env_ids, **term_cfg.params)
                else:
                    # extract last reset step for this term
                    last_triggered_step = self._reset_term_last_triggered_step_id[index][env_ids]
                    triggered_at_least_once = self._reset_term_last_triggered_once[index][env_ids]
                    # compute the steps since last reset
                    steps_since_triggered = global_env_step_count - last_triggered_step

                    # check if the term can be applied after the minimum step count between triggers has passed
                    valid_trigger = steps_since_triggered >= min_step_count
                    # check if the term has not been triggered yet (in that case, we trigger it at least once)
                    # this is usually only needed at the start of the environment
                    valid_trigger |= (last_triggered_step == 0) & ~triggered_at_least_once

                    # select the valid environment indices based on the trigger
                    if env_ids == slice(None):
                        valid_env_ids = valid_trigger.nonzero().flatten()
                    else:
                        valid_env_ids = env_ids[valid_trigger]

                    # reset the last reset step for each environment to the current env step count
                    if len(valid_env_ids) > 0:
                        self._reset_term_last_triggered_once[index][valid_env_ids] = True
                        self._reset_term_last_triggered_step_id[index][valid_env_ids] = global_env_step_count

                        # call the event term
                        term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
            else:
                # call the event term
                term_cfg.func(self._env, env_ids, **term_cfg.params)

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: EventTermCfg):
        """Sets the configuration of the specified term into the manager.

        The method finds the term by name by searching through all the modes.
        It then updates the configuration of the term with the first matching name.

        Args:
            term_name: The name of the event term.
            cfg: The configuration for the event term.

        Raises:
            ValueError: If the term name is not found.
        """
        term_found = False
        for mode, terms in self._mode_term_names.items():
            if term_name in terms:
                self._mode_term_cfgs[mode][terms.index(term_name)] = cfg
                term_found = True
                break
        if not term_found:
            raise ValueError(f"Event term '{term_name}' not found.")

    def get_term_cfg(self, term_name: str) -> EventTermCfg:
        """Gets the configuration for the specified term.

        The method finds the term by name by searching through all the modes.
        It then returns the configuration of the term with the first matching name.

        Args:
            term_name: The name of the event term.

        Returns:
            The configuration of the event term.

        Raises:
            ValueError: If the term name is not found.
        """
        for mode, terms in self._mode_term_names.items():
            if term_name in terms:
                return self._mode_term_cfgs[mode][terms.index(term_name)]
        raise ValueError(f"Event term '{term_name}' not found.")

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # buffer to store the time left for "interval" mode
        # if interval is global, then it is a single value, otherwise it is per environment
        self._interval_term_time_left: list[torch.Tensor] = list()
        # buffer to store the step count when the term was last triggered for each environment for "reset" mode
        self._reset_term_last_triggered_step_id: list[torch.Tensor] = list()
        self._reset_term_last_triggered_once: list[torch.Tensor] = list()

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
            if not isinstance(term_cfg, EventTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type EventTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )

            if term_cfg.mode != "reset" and term_cfg.min_step_count_between_reset != 0:
                omni.log.warn(
                    f"Event term '{term_name}' has 'min_step_count_between_reset' set to a non-zero value"
                    " but the mode is not 'reset'. Ignoring the 'min_step_count_between_reset' value."
                )

            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)

            # check if mode is pre-startup and scene replication is enabled
            if term_cfg.mode == "prestartup" and self._env.scene.cfg.replicate_physics:
                raise RuntimeError(
                    "Scene replication is enabled, which may affect USD-level randomization."
                    " When assets are replicated, their properties are shared across instances,"
                    " potentially leading to unintended behavior."
                    " For stable USD-level randomization, please disable scene replication"
                    " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
                )

            # for event terms with mode "prestartup", we assume a callable class term
            # can be initialized before the simulation starts.
            # this is done to ensure that the USD-level randomization is possible before the simulation starts.
            if inspect.isclass(term_cfg.func) and term_cfg.mode == "prestartup":
                omni.log.info(f"Initializing term '{term_name}' with class '{term_cfg.func.__name__}'.")
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)

            # check if mode is a new mode
            if term_cfg.mode not in self._mode_term_names:
                # add new mode
                self._mode_term_names[term_cfg.mode] = list()
                self._mode_term_cfgs[term_cfg.mode] = list()
                self._mode_class_term_cfgs[term_cfg.mode] = list()
            # add term name and parameters
            self._mode_term_names[term_cfg.mode].append(term_name)
            self._mode_term_cfgs[term_cfg.mode].append(term_cfg)

            # check if the term is a class
            if inspect.isclass(term_cfg.func):
                self._mode_class_term_cfgs[term_cfg.mode].append(term_cfg)

            # resolve the mode of the events
            # -- interval mode
            if term_cfg.mode == "interval":
                if term_cfg.interval_range_s is None:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
                    )

                # sample the time left for global
                if term_cfg.is_global_time:
                    lower, upper = term_cfg.interval_range_s
                    time_left = torch.rand(1) * (upper - lower) + lower
                    self._interval_term_time_left.append(time_left)
                else:
                    # sample the time left for each environment
                    lower, upper = term_cfg.interval_range_s
                    time_left = torch.rand(self.num_envs, device=self.device) * (upper - lower) + lower
                    self._interval_term_time_left.append(time_left)
            # -- reset mode
            elif term_cfg.mode == "reset":
                if term_cfg.min_step_count_between_reset < 0:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'reset' but 'min_step_count_between_reset' is"
                        f" negative: {term_cfg.min_step_count_between_reset}. Please provide a non-negative value."
                    )

                # initialize the current step count for each environment to zero
                step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
                self._reset_term_last_triggered_step_id.append(step_count)
                # initialize the trigger flag for each environment to zero
                no_trigger = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                self._reset_term_last_triggered_once.append(no_trigger)
