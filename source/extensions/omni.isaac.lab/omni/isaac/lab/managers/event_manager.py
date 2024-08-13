# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event manager for orchestrating operations based on different simulation events."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import carb

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import EventTermCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


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

    - "startup": Event is applied once at the beginning of the training.
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

        Note:
            For interval mode, the time step of the environment is used to determine if the event
            should be applied.

        Args:
            mode: The mode of event.
            env_ids: The indices of the environments to apply the event to.
                Defaults to None, in which case the event is applied to all environments.
            dt: The time step of the environment. This is only used for the "interval" mode.
                Defaults to None to simplify the call for other modes.
            global_env_step_count: The environment step count of the task. This is only used for the "reset" mode.
                Defaults to None to simplify the call for other modes.

        Raises:
            ValueError: If the mode is ``"interval"`` and the time step is not provided.
        """
        # check if mode is valid
        if mode not in self._mode_term_names:
            carb.log_warn(f"Event mode '{mode}' is not defined. Skipping event.")
            return
        # iterate over all the event terms
        for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):
            # resample interval if needed
            if mode == "interval":
                if dt is None:
                    raise ValueError(
                        f"Event mode '{mode}' requires the time step of the environment"
                        " to be passed to the event manager."
                    )
                if term_cfg.is_global_time:
                    # extract time left for this term
                    time_left = self._interval_mode_time_global[index]
                    # update the time left for each environment
                    time_left -= dt
                    # check if the interval has passed
                    if time_left <= 0.0:
                        lower, upper = term_cfg.interval_range_s
                        self._interval_mode_time_global[index] = torch.rand(1) * (upper - lower) + lower
                    else:
                        # no need to call func to sample
                        continue
                else:
                    # extract time left for this term
                    time_left = self._interval_mode_time_left[index]
                    # update the time left for each environment
                    time_left -= dt
                    # check if the interval has passed
                    env_ids = (time_left <= 0.0).nonzero().flatten()
                    if len(env_ids) > 0:
                        lower, upper = term_cfg.interval_range_s
                        time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (upper - lower) + lower
            # check for minimum frequency for reset
            elif mode == "reset":
                if global_env_step_count is None:
                    raise ValueError(
                        f"Event mode '{mode}' requires the step count of the environment"
                        " to be passed to the event manager."
                    )

                if env_ids is not None and len(env_ids) > 0:
                    last_reset_step = self._reset_mode_last_reset_step_count[index]
                    steps_since_last_reset = global_env_step_count - last_reset_step
                    env_ids = env_ids[steps_since_last_reset[env_ids] >= term_cfg.min_step_count_between_reset]
                    if len(env_ids) > 0:
                        last_reset_step[env_ids] = global_env_step_count
                    else:
                        # no need to call func to sample
                        continue
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
        """Prepares a list of event functions."""
        # parse remaining event terms and decimate their information
        self._mode_term_names: dict[str, list[str]] = dict()
        self._mode_term_cfgs: dict[str, list[EventTermCfg]] = dict()
        self._mode_class_term_cfgs: dict[str, list[EventTermCfg]] = dict()
        # buffer to store the time left for each environment for "interval" mode
        self._interval_mode_time_left: list[torch.Tensor] = list()
        # global timer for "interval" mode for global properties
        self._interval_mode_time_global: list[torch.Tensor] = list()
        # buffer to store the step count when reset was last performed for each environment for "reset" mode
        self._reset_mode_last_reset_step_count: list[torch.Tensor] = list()

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
                carb.log_warn(
                    f"Event term '{term_name}' has 'min_step_count_between_reset' set to a non-zero value"
                    " but the mode is not 'reset'. Ignoring the 'min_step_count_between_reset' value."
                )

            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
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
            if isinstance(term_cfg.func, ManagerTermBase):
                self._mode_class_term_cfgs[term_cfg.mode].append(term_cfg)

            # resolve the mode of the events
            if term_cfg.mode == "interval":
                if term_cfg.interval_range_s is None:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
                    )

                # sample the time left for global
                if term_cfg.is_global_time:
                    lower, upper = term_cfg.interval_range_s
                    time_left = torch.rand(1) * (upper - lower) + lower
                    self._interval_mode_time_global.append(time_left)
                else:
                    # sample the time left for each environment
                    lower, upper = term_cfg.interval_range_s
                    time_left = torch.rand(self.num_envs, device=self.device) * (upper - lower) + lower
                    self._interval_mode_time_left.append(time_left)

            elif term_cfg.mode == "reset":
                step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
                self._reset_mode_last_reset_step_count.append(step_count)
