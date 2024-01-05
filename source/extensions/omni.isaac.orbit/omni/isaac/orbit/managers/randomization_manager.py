# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Randomization manager for randomizing different elements in the scene."""

from __future__ import annotations

import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Sequence

import carb

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RandomizationTermCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


class RandomizationManager(ManagerBase):
    """Manager for randomizing different elements in the scene.

    The randomization manager applies randomization to any instance in the scene. For example, changing the
    masses of objects or their friction coefficients, or applying random pushes to the robot. The user can
    specify several modes of randomization to specialize the behavior based on when to apply the randomization.

    The randomization terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each randomization term should instantiate the :class:`RandomizationTermCfg` class.

    Randomization terms can be grouped by their mode. The mode is a user-defined string that specifies when
    the randomization term should be applied. This provides the user complete control over when randomization
    terms should be applied.

    For a typical training process, you may want to randomize in the following modes:

    - "startup": Randomization term is applied once at the beginning of the training.
    - "reset": Randomization is applied at every reset.
    - "interval": Randomization is applied at pre-specified intervals of time.

    However, you can also define your own modes and use them in the training process as you see fit.

    .. note::

        The mode ``"interval"`` is the only mode that is handled by the manager itself which is based on
        the environment's time step.

    """

    _env: RLTaskEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: RLTaskEnv):
        """Initialize the randomization manager.

        Args:
            cfg: A configuration object or dictionary (``dict[str, RandomizationTermCfg]``).
            env: An environment object.
        """
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for randomization manager."""
        msg = f"<RandomizationManager> contains {len(self._mode_term_names)} active terms.\n"

        # add info on each mode
        for mode in self._mode_term_names:
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Randomization Terms in Mode: '{mode}'"
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
        """Name of active randomization terms.

        The keys are the modes of randomization and the values are the names of the randomization terms.
        """
        return self._mode_term_names

    @property
    def available_modes(self) -> list[str]:
        """Modes of randomization."""
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

    def randomize(self, mode: str, env_ids: Sequence[int] | None = None, dt: float | None = None):
        """Calls each randomization term in the specified mode.

        Note:
            For interval mode, the time step of the environment is used to determine if the randomization should be
            applied. If the time step is not constant, the user should pass the time step to this function.

        Args:
            mode: The mode of randomization.
            env_ids: The indices of the environments to apply randomization to.
                Defaults to None, in which case the randomization is applied to all environments.
            dt: The time step of the environment. This is only used for the "interval" mode.
                Defaults to None, in which case the randomization is not applied.

        Raises:
            ValueError: If the mode is ``"interval"`` and the time step is not provided.
        """
        # check if mode is valid
        if mode not in self._mode_term_names:
            carb.log_warn(f"Randomization mode '{mode}' is not defined. Skipping randomization.")
            return
        # iterate over all the randomization terms
        for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):
            # resample interval if needed
            if mode == "interval":
                if dt is None:
                    raise ValueError(
                        f"Randomization mode '{mode}' requires the time step of the environment"
                        " to be passed to the randomization manager."
                    )
                # extract time left for this term
                time_left = self._interval_mode_time_left[index]
                # update the time left for each environment
                time_left -= dt
                # check if the interval has passed
                env_ids = (time_left <= 0.0).nonzero().flatten()
                if len(env_ids) > 0:
                    lower, upper = term_cfg.interval_range_s
                    time_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (upper - lower) + lower
            # call the randomization term
            term_cfg.func(self._env, env_ids, **term_cfg.params)

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: RandomizationTermCfg):
        """Sets the configuration of the specified term into the manager.

        The method finds the term by name by searching through all the modes.
        It then updates the configuration of the term with the first matching name.

        Args:
            term_name: The name of the randomization term.
            cfg: The configuration for the randomization term.

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
            raise ValueError(f"Randomization term '{term_name}' not found.")

    def get_term_cfg(self, term_name: str) -> RandomizationTermCfg:
        """Gets the configuration for the specified term.

        The method finds the term by name by searching through all the modes.
        It then returns the configuration of the term with the first matching name.

        Args:
            term_name: The name of the randomization term.

        Returns:
            The configuration of the randomization term.

        Raises:
            ValueError: If the term name is not found.
        """
        for mode, terms in self._mode_term_names.items():
            if term_name in terms:
                return self._mode_term_cfgs[mode][terms.index(term_name)]
        raise ValueError(f"Randomization term '{term_name}' not found.")

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of randomization functions."""
        # parse remaining randomization terms and decimate their information
        self._mode_term_names: dict[str, list[str]] = dict()
        self._mode_term_cfgs: dict[str, list[RandomizationTermCfg]] = dict()
        self._mode_class_term_cfgs: dict[str, list[RandomizationTermCfg]] = dict()
        # buffer to store the time left for each environment for "interval" mode
        self._interval_mode_time_left: list[torch.Tensor] = list()

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
            if not isinstance(term_cfg, RandomizationTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type RandomizationTermCfg."
                    f" Received: '{type(term_cfg)}'."
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

            # resolve the mode of randomization
            if term_cfg.mode == "interval":
                if term_cfg.interval_range_s is None:
                    raise ValueError(
                        f"Randomization term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
                    )
                # sample the time left for each environment
                lower, upper = term_cfg.interval_range_s
                time_left = torch.rand(self.num_envs, device=self.device) * (upper - lower) + lower
                self._interval_mode_time_left.append(time_left)
