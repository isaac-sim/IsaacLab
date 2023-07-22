# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Randomization manager for randomizing different elements in the scene."""

import logging
import torch
from prettytable import PrettyTable
from typing import Dict, List, Optional, Sequence

from .manager_base import ManagerBase
from .manager_cfg import RandomizationTermCfg


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

    def __init__(self, cfg: object, env: object):
        """Initialize the randomization manager.

        Args:
            cfg (object): A configuration object or dictionary (``dict[str, RandomizationTermCfg]``).
            env (object): An environment object.
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
    def dt(self) -> float:
        """The environment time-step (in seconds)."""
        return self._env.dt

    @property
    def active_terms(self) -> Dict[str, List[str]]:
        """Name of active randomization terms."""
        return self._mode_term_names

    """
    Operations.
    """

    def randomize(self, mode: str, env_ids: Optional[Sequence[int]] = None, dt: Optional[float] = None):
        """Calls each randomization term in the specified mode.

        Note:
            For interval mode, the time step of the environment is used to determine if the randomization should be
            applied. If the time step is not constant, the user should pass the time step to this function.

        Args:
            mode (str): The mode of randomization.
            env_ids (Optional[Sequence[int]]): The indices of the environments to apply randomization to.
                Defaults to None, in which case the randomization is applied to all environments.
            dt (Optional[float]): The time step of the environment. Defaults to None, in which case the time
                step of the environment is used.
        """
        # check if mode is valid
        if mode not in self._mode_term_names:
            logging.warning(f"Randomization mode '{mode}' is not defined. Skipping randomization.")
            return
        # iterate over all the randomization terms
        for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):
            # resample interval if needed
            if mode == "interval":
                if dt is None:
                    dt = self.dt
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
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of randomization functions."""
        # parse remaining randomization terms and decimate their information
        self._mode_term_names: Dict[str, List[str]] = dict()
        self._mode_term_cfgs: Dict[str, List[RandomizationTermCfg]] = dict()
        # buffer to store the time left for each environment for "interval" mode
        self._interval_mode_time_left: List[torch.Tensor] = list()

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
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
            # check if mode is a new mode
            if term_cfg.mode not in self._mode_term_names:
                # add new mode
                self._mode_term_names[term_cfg.mode] = list()
                self._mode_term_cfgs[term_cfg.mode] = list()
            # add term name and parameters
            self._mode_term_names[term_cfg.mode].append(term_name)
            self._mode_term_cfgs[term_cfg.mode].append(term_cfg)

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
