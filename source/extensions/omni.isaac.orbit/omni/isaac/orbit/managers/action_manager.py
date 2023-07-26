# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Action manager for processing actions sent to the environment."""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from prettytable import PrettyTable

from .manager_base import ManagerBase
from .manager_cfg import ActionTermCfg


class ActionTerm(ABC):
    """Base class for action terms."""

    # TODO: Should this be here or a property?
    # Are they even exposed to the user?
    raw_actions: torch.Tensor
    processed_actions: torch.Tensor

    def __init__(self, cfg: ActionTermCfg, env: object):
        """Initialize the action term.

        Args:
            cfg (ActionTermCfg): The configuration object.
            env (object): The environment instance.
        """
        # store the inputs
        self._cfg = cfg
        self._env = env
        # parse config to obtain asset to which the term is applied
        self._asset = getattr(env, cfg.asset_name)

    """
    Properties.
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
    def action_dim(self) -> int:
        """Dimension of the action term."""
        raise NotImplementedError

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions (torch.Tensor): The actions to process.
        """
        # TODO: Why not make this an abstract method?
        # This one line can be implemented in the child class.
        self.raw_actions[:] = actions

    @abstractmethod
    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """
        raise NotImplementedError


class ActionManager(ManagerBase):
    """Manager for processing and applying actions for a given world.

    The action manager handles the interpretation and application of user-defined
    actions on a given world. It is comprised of different action terms that decide
    the dimension of the expected actions.

    The action manager performs operations at two stages:

    * processing of actions: It splits the input actions to each term and performs any
      pre-processing needed. This should be called once at every environment step.
    * apply actions: This operation typically sets the processed actions into the assets in the
      scene (such as robots). It should be called before every simulation step.
    """

    def __init__(self, cfg: object, env: object):
        """Initialize the action manager.

        Args:
            cfg (object): The configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env (object): The environment instance.
        """
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        # add info on each term
        for index, (name, term) in enumerate(zip(self._term_names, self._terms)):
            table.add_row([index, name, term.action_dim])
        # convert table to string
        msg += table.get_string()

        return msg

    """
    Properties.
    """

    @property
    def total_action_dim(self) -> int:
        """Total dimension of actions."""
        return sum(self.action_term_dim)

    @property
    def active_terms(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [term.action_dim for term in self._terms]

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            actions (torch.Tensor): The actions to process.
        """
        # check if action dimension is valid
        if self.total_action_dim != actions.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {actions.shape[1]}")

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms:
            term_actions = actions[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)

    def apply_actions(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms:
            term.apply_actions()

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of action terms."""
        # parse action terms from the config
        self._term_names: list[str] = list()
        self._terms: list[ActionTerm] = list()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            # check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg. Received '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.cls(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms.append(term)
