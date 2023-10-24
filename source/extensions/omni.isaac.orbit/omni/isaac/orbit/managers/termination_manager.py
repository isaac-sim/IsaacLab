# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Termination manager for computing done signals for a given world."""

from __future__ import annotations

import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Sequence

from .manager_base import ManagerBase
from .manager_cfg import TerminationTermCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLEnv


class TerminationManager(ManagerBase):
    """Manager for computing done signals for a given world.

    The termination manager computes the termination signal (also called dones) as a combination
    of termination terms. Each termination term is a function which takes the environment as an
    argument and returns a boolean tensor of shape ``(num_envs,)``. The termination manager
    computes the termination signal as the union (logical or) of all the termination terms.

    The termination terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each termination term should instantiate the :class:`TerminationTermCfg` class.
    """

    _env: RLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: RLEnv):
        """Initializes the termination manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, TerminationTermCfg]``).
            env: An environment object.
        """
        super().__init__(cfg, env)
        # prepare extra info to store individual termination term information
        self._episode_dones = dict()
        for term_name in self._term_names:
            self._episode_dones[term_name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # create buffer for managing termination per environment
        self._done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._time_out_buf = torch.zeros_like(self._done_buf)

    def __str__(self) -> str:
        """Returns: A string representation for termination manager."""
        msg = f"<TerminationManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Termination Terms"
        table.field_names = ["Index", "Name", "Time Out"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, term_cfg.time_out])
        # convert table to string
        msg += table.get_string()

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active termination terms."""
        return self._term_names

    @property
    def dones(self) -> torch.Tensor:
        """The net termination signal. Shape is ``(num_envs,)``."""
        return self._done_buf

    @property
    def time_outs(self) -> torch.Tensor:
        """The timeout signal. Shape is ``(num_envs,)``."""
        return self._time_out_buf

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic counts of individual termination terms.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # add to episode dict
        extras = {}
        for key in self._episode_dones.keys():
            # store information
            extras["Episode Termination/" + key] = torch.count_nonzero(self._episode_dones[key][env_ids]).item()
            # reset episode dones
            self._episode_dones[key][env_ids] = False
        return extras

    def compute(self) -> torch.Tensor:
        """Computes the termination signal as union of individual terms.

        This function calls each termination term managed by the class and performs a logical OR operation
        to compute the net termination signal.

        Returns:
            The combined termination signal of shape ``(num_envs,)``.
        """
        # reset computation
        self._done_buf[:] = False
        self._time_out_buf[:] = False
        # iterate over all the termination terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            value = term_cfg.func(self._env, **term_cfg.params)
            # update total termination
            self._done_buf |= value
            # store timeout signal separately
            if term_cfg.time_out:
                self._time_out_buf |= value
            # add to episode dones
            self._episode_dones[name] |= value
        # return termination signal
        return self._done_buf

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of termination functions."""
        # parse remaining termination terms and decimate their information
        self._term_names: list[str] = list()
        self._term_cfgs: list[TerminationTermCfg] = list()

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
            if not isinstance(term_cfg, TerminationTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type TerminationTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
