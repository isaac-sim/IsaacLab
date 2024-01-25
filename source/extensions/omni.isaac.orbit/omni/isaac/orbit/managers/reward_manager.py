# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward manager for computing reward signals for a given world."""

from __future__ import annotations

import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Sequence

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RewardTermCfg, RewardGroupCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


DEFAULT_GROUP_NAME = "reward"


class RewardManager(ManagerBase):
    """Manager for computing reward signals for a given world.

    The reward manager computes the total reward as a sum of the weighted reward terms. The reward
    terms are parsed from a nested config class containing the reward manger's settings and reward
    terms configuration.

    The reward terms are parsed from a config class containing the manager's settings and each term's
    parameters.

    Rewards are organized into groups, for multi-critic or CMDP use-cases.
    Each rewards group shouuld inherit from the :class:`RewardGroupCfg` class.
    Within each group, each reward term should inherit from the :class:`RewardTermCfg` class.

    .. note::

        The reward manager multiplies the reward term's ``weight``  with the time-step interval ``dt``
        of the environment. This is done to ensure that the computed reward terms are balanced with
        respect to the chosen time-step interval in the environment.

    """

    _env: RLTaskEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: RLTaskEnv):
        """Initialize the reward manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RewardGroupCfg]``).
            env: The environment instance.
        """
        # Variable to track whether we have reward groups or not.
        # Needs to be set before we call super().__init__ because it's needed in prepare_terms.
        self.no_group = None
        super().__init__(cfg, env)

        # Allocate storage for reward terms.
        self._episode_sums = dict()
        self._reward_buf = {}
        self._term_names_flat = []  # flat list of all term names
        for group_name, group_term_names in self._group_term_names.items():
            for term_name in group_term_names:
                sum_term_name = (
                    term_name if self.no_group else f"{group_name}/{term_name}"
                )
                self._episode_sums[sum_term_name] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device
                )

                self._term_names_flat.append(sum_term_name)

            # create buffer for managing reward per environment
            self._reward_buf[group_name] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        # Get number of reward terms.
        msg = f"<RewardManager> contains {len(self._term_names_flat)} active terms.\n"

        # create table for term information
        for group_name in self._group_term_names.keys():
            table = PrettyTable()
            table.title = "Active Reward Terms In Group: " + group_name
            table.field_names = ["Index", "Name", "Weight"]
            # set alignment of table columns
            table.align["Name"] = "l"
            table.align["Weight"] = "r"
            # add info on each term
            for index, (name, term_cfg) in enumerate(
                zip(
                    self._group_term_names[group_name],
                    self._group_term_cfgs[group_name],
                )
            ):
                table.add_row([index, name, term_cfg.weight])
            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active reward terms."""
        return self._term_names_flat

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual reward terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual reward terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for key in self._episode_sums.keys():
            # store information
            # r_1 + r_2 + ... + r_n
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode Reward/" + key] = (
                episodic_sum_avg / self._env.max_episode_length_s
            )
            # reset episodic sum
            self._episode_sums[key][env_ids] = 0.0
        # reset all the reward terms
        for group_cfg in self._group_class_term_cfgs.values():
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self, dt: float) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        for key in self._reward_buf.keys():
            self._reward_buf[key][:] = 0.0
        # iterate over all reward terms of all groups
        for group_name in self._group_term_names.keys():
            # iterate over all the reward terms
            for term_name, term_cfg in zip(
                self._group_term_names[group_name], self._group_term_cfgs[group_name]
            ):
                # skip if weight is zero (kind of a micro-optimization)
                if term_cfg.weight == 0.0:
                    continue
                # compute term's value
                value = (
                    term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
                )
                # update total reward
                self._reward_buf[group_name] += value
                # update episodic sum
                name = term_name if self.no_group else f"{group_name}/{term_name}"
                self._episode_sums[name] += value

        # Return only Tensor if config has no groups.
        if self.no_group:
            return self._reward_buf[DEFAULT_GROUP_NAME]
        else:
            return self._reward_buf

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: RewardTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the reward term.
            cfg: The configuration for the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        # Split term_name at '/' if it has one.
        if "/" in term_name:
            group_name, term_name = term_name.split("/")
        else:
            group_name = DEFAULT_GROUP_NAME

        if group_name not in self._group_term_names:
            raise ValueError(f"Reward group '{group_name}' not found.")
        if term_name not in self._group_term_names[group_name]:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # set the configuration
        self._group_term_cfgs[group_name][
            self._group_term_names[group_name].index(term_name)
        ] = cfg

    def get_term_cfg(self, term_name: str) -> RewardTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the reward term.

        Returns:
            The configuration of the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        # Split term_name at '/' if it has one.
        if "/" in term_name:
            group_name, term_name = term_name.split("/")
        else:
            group_name = DEFAULT_GROUP_NAME

        if group_name not in self._group_term_names:
            raise ValueError(f"Reward group '{group_name}' not found.")
        if term_name not in self._group_term_names[group_name]:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # return the configuration
        return self._group_term_cfgs[group_name][
            self._group_term_names[group_name].index(term_name)
        ]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of reward functions."""

        self._group_term_names: dict[str, list[str]] = dict()
        self._group_term_cfgs: dict[str, list[RewardTermCfg]] = dict()
        self._group_class_term_cfgs: dict[str, list[RewardTermCfg]] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        # Check whether we have a group or not and fail if we have a mix.
        for name, cfg in cfg_items:
            # check for non config
            if cfg is None:
                continue
            if isinstance(cfg, RewardTermCfg):
                if self.no_group is None:
                    self.no_group = True
                elif self.no_group is False:
                    raise ValueError("Cannot mix reward groups with reward terms.")
            else:
                if self.no_group is None:
                    self.no_group = False
                elif self.no_group is True:
                    raise ValueError("Cannot mix reward groups with reward terms.")

        # Make a group if we do not have one.
        if self.no_group:
            cfg_items = {DEFAULT_GROUP_NAME: dict(cfg_items)}.items()

        # iterate over all the groups
        for group_name, group_cfg in cfg_items:
            self._group_term_names[group_name] = list()
            self._group_term_cfgs[group_name] = list()
            self._group_class_term_cfgs[group_name] = list()

            # Make group config a list if it is not.
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items()

            # Iterate over all the terms in the group
            for term_name, term_cfg in group_cfg_items:
                # check for non config
                if term_cfg is None:
                    continue
                # check for valid config type
                if not isinstance(term_cfg, RewardTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type RewardTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # check for valid weight type
                if not isinstance(term_cfg.weight, (float, int)):
                    raise TypeError(
                        f"Weight for the term '{term_name}' is not of type float or int."
                        f" Received: '{type(term_cfg.weight)}'."
                    )
                # resolve common terms in the config
                self._resolve_common_term_cfg(
                    f"{group_name}/{term_name}", term_cfg, min_argc=1
                )
                # add term config to list
                self._group_term_names[group_name].append(term_name)
                self._group_term_cfgs[group_name].append(term_cfg)
                # add term to separate list if term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_class_term_cfgs[group_name].append(term_cfg)
                    # call reset on the term
                    term_cfg.func.reset()
