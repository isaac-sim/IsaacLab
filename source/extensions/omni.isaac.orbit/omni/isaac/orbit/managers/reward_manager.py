# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Reward manager for computing reward signals for a given world."""

import torch
from prettytable import PrettyTable
from typing import Dict, List, Optional, Sequence

from .manager_base import ManagerBase
from .manager_cfg import RewardTermCfg


class RewardManager(ManagerBase):
    """Manager for computing reward signals for a given world.

    The reward manager computes the total reward as a sum of the weighted reward terms. The reward
    terms are parsed from a nested config class containing the reward manger's settings and reward
    terms configuration.

    The reward terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each reward term should instantiate the :class:`RewardTermCfg` class.

    .. note::

        The reward manager multiplies the reward term's ``weight``  with the time-step interval ``dt``
        of the environment. This is done to ensure that the computed reward terms are balanced with
        respect to the chosen time-step interval in the environment.

    """

    def __init__(self, cfg: object, env: object):
        """Initialize the reward manager.

        Args:
            cfg (object): The configuration object or dictionary (``dict[str, RewardTermCfg]``).
            env (object): The environment instance.
        """
        super().__init__(cfg, env)
        # prepare extra info to store individual reward term information
        self._episode_sums = dict()
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # create buffer for managing reward per environment
        self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Reward Terms"
        table.field_names = ["Index", "Name", "Weight"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, term_cfg.weight])
        # convert table to string
        msg += table.get_string()

        return msg

    """
    Properties.
    """

    @property
    def dt(self) -> float:
        """The environment time-step (in seconds)."""
        return self._env.dt

    @property
    def active_terms(self) -> List[str]:
        """Name of active reward terms."""
        return self._term_names

    """
    Operations.
    """

    def log_info(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, torch.Tensor]:
        """Returns the episodic sum of individual reward terms.

        Args:
            env_ids (Sequence[int], optional): The environment ids for which the episodic sum of
                individual reward terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = ...
        # store information
        extras = {}
        for key in self._episode_sums.keys():
            extras["Episode Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / (
                self._env.max_episode_length * self.dt
            )  # FIXME
            self._episode_sums[key][env_ids] = 0.0
        return extras

    def compute(self) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Returns:
            torch.Tensor: The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        # iterate over all the reward terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value

        return self._reward_buf

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of reward functions."""
        # parse remaining reward terms and decimate their information
        self._term_names: List[str] = list()
        self._term_cfgs: List[RewardTermCfg] = list()

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
            if not isinstance(term_cfg, RewardTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type RewardTermCfg. Received '{type(term_cfg)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # remove zero scales and multiply non-zero ones by dt
            # note: we multiply weights by dt to make them agnostic to control decimation
            if term_cfg.weight == 0:
                continue
            term_cfg.weight *= self.dt
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
