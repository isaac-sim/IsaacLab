# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum manager for updating environment quantities subject to a training curriculum."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

from ..utils import env_selection_kwargs
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import CurriculumTermCfg

if TYPE_CHECKING:
    from ..envs import ManagerBasedRLEnv


class CurriculumManager(ManagerBase):
    """Manager to implement and execute specific curricula.

    The curriculum manager updates various quantities of the environment subject to a training curriculum by
    calling a list of terms. These help stabilize learning by progressively making the learning tasks harder
    as the agent improves.

    The curriculum terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each curriculum term should instantiate the :class:`CurriculumTermCfg` class.
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, CurriculumTermCfg]``)
            env: An environment object.

        Raises:
            TypeError: If curriculum term is not of type :class:`CurriculumTermCfg`.
            ValueError: If curriculum term configuration does not satisfy its function signature.
        """
        # create buffers to parse and store terms
        self._term_names: list[str] = []
        self._term_cfgs: list[CurriculumTermCfg] = []
        self._class_term_cfgs: list[CurriculumTermCfg] = []

        # call the base class constructor (this will parse the terms config)
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

    def reset(
        self, env_ids: Sequence[int] | None = None, env_mask: torch.Tensor | None = None
    ) -> dict[str, float | torch.Tensor]:
        """Returns the current state of individual curriculum terms.

        Note:
            The logged state covers all the terms regardless of the selected environments. The selection only
            resets the class terms.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all environments are considered.
            env_mask: Boolean mask of the environments to reset. Shape is (num_envs,). Takes precedence over
                ``env_ids``.

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
                            value = value.detach().clone()
                        extras[f"Curriculum/{term_name}/{key}"] = value
                else:
                    # log directly if not a dict
                    if isinstance(term_state, torch.Tensor):
                        term_state = term_state.detach().clone()
                    extras[f"Curriculum/{term_name}"] = term_state
        # reset all the curriculum terms
        env_mask, env_ids = self._env_selection(env_ids, env_mask)
        for term_cfg in self._class_term_cfgs:
            self._reset_term(term_cfg.func, env_mask, env_ids)
        return extras

    def compute(self, env_ids: Sequence[int] | None = None, env_mask: torch.Tensor | None = None):
        """Update the curriculum terms.

        This function calls each curriculum term managed by the class. Terms that take ``env_mask`` are called
        on every update; terms that take indices are skipped when no environment is selected.

        Args:
            env_ids: The list of environment IDs to update.
                If None, all the environments are updated. Defaults to None.
            env_mask: Boolean mask of the environments to update. Shape is (num_envs,). Takes precedence over
                ``env_ids``.
        """
        env_mask, env_ids = self._env_selection(env_ids, env_mask)
        # iterate over all the curriculum terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            selection = env_selection_kwargs(term_cfg.func, env_mask, env_ids)
            if selection is not None:
                self._curriculum_state[name] = term_cfg.func(self._env, *selection.values(), **term_cfg.params)

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """

        terms = []

        for term_name, term_state in self._curriculum_state.items():
            if term_state is not None:
                # deal with dict
                data = []

                if isinstance(term_state, dict):
                    # each key is a separate state to log
                    for key, value in term_state.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        data.append(value)
                else:
                    # log directly if not a dict
                    if isinstance(term_state, torch.Tensor):
                        term_state = term_state.item()
                    data.append(term_state)
                terms.append((term_name, data))

        return terms

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
