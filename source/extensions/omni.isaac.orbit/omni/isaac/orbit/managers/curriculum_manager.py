# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum manager for updating environment quantities subject to a training curriculum."""

from __future__ import annotations

import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Sequence

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import CurriculumTermCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


class CurriculumManager(ManagerBase):
    """Manager to implement and execute specific curricula.

    The curriculum manager updates various quantities of the environment subject to a training curriculum by
    calling a list of terms. These help stabilize learning by progressively making the learning tasks harder
    as the agent improves.

    The curriculum terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each curriculum term should instantiate the :class:`CurriculumTermCfg` class.
    """

    _env: RLTaskEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: RLTaskEnv):
        """Initialize the manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, CurriculumTermCfg]``)
            env: An environment object.

        Raises:
            TypeError: If curriculum term is not of type :class:`CurriculumTermCfg`.
            ValueError: If curriculum term configuration does not satisfy its function signature.
        """
        super().__init__(cfg, env)
        # prepare logging
        self._curriculum_state = dict()
        for term_name in self._term_names:
            self._curriculum_state[term_name] = None

    def __str__(self) -> str:
        """Returns: A string representation for curriculum manager."""
        msg = f"<CurriculumManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Curriculum Terms"
        table.field_names = ["Index", "Name"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, name in enumerate(self._term_names):
            table.add_row([index, name])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active curriculum terms."""
        return self._term_names

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Returns the current state of individual curriculum terms.

        Note:
            This function does not use the environment indices :attr:`env_ids`
            and logs the state of all the terms. The argument is only present
            to maintain consistency with other classes.

        Returns:
            Dictionary of curriculum terms and their states.
        """
        extras = {}
        for term_name, term_state in self._curriculum_state.items():
            if term_state is not None:
                # deal with dict
                if isinstance(term_state, dict):
                    # each key is a separate state to log
                    for key, value in term_state.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        extras[f"Curriculum/{term_name}/{key}"] = value
                else:
                    # log directly if not a dict
                    if isinstance(term_state, torch.Tensor):
                        term_state = term_state.item()
                    extras[f"Curriculum/{term_name}"] = term_state
        # reset all the curriculum terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self, env_ids: Sequence[int] | None = None):
        """Update the curriculum terms.

        This function calls each curriculum term managed by the class.

        Args:
            env_ids: The list of environment IDs to update.
                If None, all the environments are updated. Defaults to None.
        """
        # resolve environment indices
        if env_ids is None:
            env_ids = slice(None)
        # iterate over all the curriculum terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            state = term_cfg.func(self._env, env_ids, **term_cfg.params)
            self._curriculum_state[name] = state

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # parse remaining curriculum terms and decimate their information
        self._term_names: list[str] = list()
        self._term_cfgs: list[CurriculumTermCfg] = list()
        self._class_term_cfgs: list[CurriculumTermCfg] = list()

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
            # check if the term is a valid term config
            if not isinstance(term_cfg, CurriculumTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type CurriculumTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
            # add name and config to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)
