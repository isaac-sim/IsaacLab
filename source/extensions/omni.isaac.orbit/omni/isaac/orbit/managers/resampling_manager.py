# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resampling manager for resampling commands."""

from __future__ import annotations

import torch
from abc import abstractmethod
from typing import TYPE_CHECKING, Sequence

import carb
import omni.kit.app

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ResamplingTermCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


class ResamplingTerm(ManagerTermBase):
    """The base class for implementing a resampling term.

    A resampling term is used to decide whether a command should be resampled or not,
    every time a command is updated. The terms are called by the resampling manager,
    which in turn is called by the command term which owns it.
    """

    def __init__(self, cfg, env):
        """Initialize the resampling term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        # initialize the base class
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation of the resampling term config."""
        return ""

    @abstractmethod
    def compute(self, dt: float):
        """Compute the environment ids to be resampled.

        Args:
            dt: The time step.
        """
        # compute the resampling term
        raise NotImplementedError

    def reset(self, env_ids: Sequence[int]):
        """Reset the resampling term.

        Some resamplig terms need to know when their commands are resampled
        (e.g. th FixedFrequency term needs to reset its time left counter).
        Since a command term can have multiple resampling terms, it is not
        possible to know for sure when the command is resampled. This function is
        called by the command term when it resamples to ensure that all resampling
        terms are properly reset.

        Args:
            env_ids: The environment ids to be reset.
        """
        pass


class ResamplingManager(ManagerBase):
    """Manager for resampling commands.

    The resampling manager is used to determine whether a command should
    be resampled or not. It will be owned by a CommandTerm, which calls
    the resampling manager every time it computes a command, to decide
    which environment ids should be resampled.

    The manager consists of multiple reward terms. Every reward term is
    called to get environment ids to be resampled. The union of these
    ids are resampled by the CommandTerm.

    Args:
        cfg: The configuration object.
        env: The environment instance.
    """

    _env: RLTaskEnv
    """The environment instance."""

    def __init__(self, cfg, env):
        """Initialize the resampling manager.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """

        # initialize the base class
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation of the resampling manager."""
        msg = "ResamplingManager:\n"
        msg += f"\tNumber of terms: {len(self._terms)}\n"
        for index, (name, term) in enumerate(self._terms.items()):
            msg += f"\t{index}\t{name}: {term.__class__.__name__}\n{term}\n"
        return msg

    def compute(self, dt: float):
        """Compute the environment ids to be resampled.

        Calls all reward terms and returns the union of the environment ids.

        Args:
            dt: The time step since last computing the resampling ids.
        """
        env_ids = []
        for term in self._terms.values():
            env_ids.append(term.compute(dt).view(-1))

        return torch.unique(torch.cat(env_ids))

    def reset(self, env_ids: Sequence[int] | None = None):
        """
        Resets all resampling terms.

        Args:
            env_ids: The environment ids to be reset. If None, all the environments
                are reset.
        """
        if env_ids is None:
            env_ids = slice(None)

        for term in self._terms.values():
            term.reset(env_ids)

    def _prepare_terms(self):
        """Prepares a list of resampling terms."""
        # parse command terms from the config
        self._terms: dict[str, ResamplingTerm] = dict()

        if self.cfg is None:
            carb.log_warn("Got no resampling terms. Some commands might not be resampled.")
            return

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
            if not isinstance(term_cfg, ResamplingTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ResamplingTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # add class to dict
            self._terms[term_name] = term

    @property
    def active_terms(self) -> list[str]:
        """Name of active command terms."""
        return list(self._terms.keys())
