# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward manager for computing reward signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RewardGroupCfg, RewardTermCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


DEFAULT_GROUP_NAME = "reward"
"""Default group name for backwards compatibility.

If the user does not define any groups, the reward terms are added to this default group.
"""


class RewardManager(ManagerBase):
    """Manager for computing reward signals for a given world.

    Rewards are organized into groups, where the total reward is computed independently for each group. Splitting into
    groups is useful for multi-agent scenarios, where each agent has its own set of rewards or to compute an additional
    oracle reward signal.

    Each reward group should inherit from the :class:`RewardGroupCfg` class. Within each group, each
    reward term should instantiate the :class:`RewardTermCfg` class. The reward manager computes the total
    reward for each group as a sum of the weighted reward terms.

    .. note::

        The reward manager multiplies the reward term's ``weight``  with the time-step interval ``dt``
        of the environment. This is done to ensure that the computed reward terms are balanced with
        respect to the chosen time-step interval in the environment.

        For backwards compatibility, the reward manager also supports the old configuration format without
        groups.

    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the reward manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RewardGroupCfg]``).
            env: The environment instance.
        """

        super().__init__(cfg, env)
        # prepare extra info to store individual reward term information
        self._episode_sums = dict()
        self._reward_buf = dict()
        for group_name, group_term_names in self._group_term_names.items():
            self._episode_sums[group_name] = dict()
            # create the total reward buffer for each group
            self._reward_buf[group_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for term_name in group_term_names:
                self._episode_sums[group_name][term_name] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device
                )

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<RewardManager> contains {len(self._group_term_names)} groups.\n"

        # add info for each group
        for group_name in self._group_term_names.keys():
            # create table for term information
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
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active reward terms in each group."""
        return self._group_term_names

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets every reward group and returns the episodic sum of individual reward terms for each group.

        Args:
            env_ids: The environment ids to reset.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # store information
        extras = {}
        for group_name in self._group_term_names.keys():
            extras.update(self.reset_group(group_name, env_ids))
        # return logged information
        return extras

    def reset_group(self, group_name: str, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the reward group and returns the episodic sum of individual reward terms in the group.

        Args:
            group_name: The name of the group to reset.
            env_ids: The environment ids to reset.

        Returns:
            Dictionary of episodic sum of individual reward terms.

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """

        # check if group name is valid
        if group_name not in self._group_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the reward manager."
                f" Available groups are: {list(self._group_term_names.keys())}"
            )

        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for key in self._episode_sums[group_name].keys():
            # store information
            # r_1 + r_2 + ... + r_n
            episodic_sum_avg = torch.mean(self._episode_sums[group_name][key][env_ids])
            name = f"Episode_Reward/{group_name}/{key}" if self.has_groups else f"Episode_Reward/{key}"
            extras[name] = episodic_sum_avg / self._env.max_episode_length_s
            # reset episodic sum
            self._episode_sums[group_name][key][env_ids] = 0.0
        # reset all the reward terms
        for term_cfg in self._group_class_term_cfgs[group_name]:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self, dt: float) -> dict[str, torch.Tensor]:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # iterate over all the reward terms
        for group_name in self._group_term_names.keys():
            self.compute_group(group_name, dt)
        return self._reward_buf if self.has_groups else self._reward_buf[DEFAULT_GROUP_NAME]

    def compute_group(self, group_name: str, dt: float) -> torch.Tensor:
        """Computes the weighted sum of rewards for a given group.

        The reward for a given group are computed by calling the registered functions for each
        term in the group.

        Args:
            group_name: The name of the group for which to compute the rewards.
            dt: The time-step interval of the environment.

        Returns:
            The weighted sum of rewards for the group of shape (num_envs,).

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """
        # check if group name is valid
        if group_name not in self._group_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the reward manager."
                f" Available groups are: {list(self._group_term_names.keys())}"
            )
        # reset computation
        self._reward_buf[group_name][:] = 0.0
        for term_name, term_cfg in zip(self._group_term_names[group_name], self._group_term_cfgs[group_name]):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            self._reward_buf[group_name] += value
            # update episodic sum
            self._episode_sums[group_name][term_name] += value
        return self._reward_buf[group_name]

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: RewardTermCfg, group_name: str = DEFAULT_GROUP_NAME):
        """Sets the configuration of the specified term into the manager.

        Args:
            group_name: The name of the reward group.
            term_name: The name of the reward term.
            cfg: The configuration for the reward term.

        Raises:
            ValueError: If the group or term name is not found.
        """
        if group_name not in self._group_term_names:
            raise ValueError(f"Reward group '{group_name}' not found.")

        if term_name not in self._group_term_names[group_name]:
            raise ValueError(f"Reward term '{term_name}' not found.")

        self._group_term_cfgs[group_name][self._group_term_names[group_name].index(term_name)] = cfg

    def get_term_cfg(self, term_name: str, group_name: str = DEFAULT_GROUP_NAME) -> RewardTermCfg:
        """Gets the configuration for the specified term.

        Args:
            group_name: The name of the reward group.
            term_name: The name of the reward term.

        Returns:
            The configuration of the reward term.

        Raises:
            ValueError: If the group or term name is not found.
        """
        if group_name not in self._group_term_names:
            raise ValueError(f"Reward group '{group_name}' not found.")

        if term_name not in self._group_term_names[group_name]:
            raise ValueError(f"Reward term '{term_name}' not found.")

        return self._group_term_cfgs[group_name][self._group_term_names[group_name].index(term_name)]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of reward functions."""
        # create buffers to store information for each reward group
        self._group_term_names: dict[str, list[str]] = dict()
        self._group_term_cfgs: dict[str, list[RewardTermCfg]] = dict()
        self._group_class_term_cfgs: dict[str, list[RewardTermCfg]] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        # ensure backwards compatibility. If the user has not defined groups, create a default group.
        self.has_groups = all(isinstance(cfg, RewardGroupCfg) for _, cfg in cfg_items)
        if not self.has_groups:
            group_cfg_item = RewardGroupCfg()
            for name, cfg in cfg_items:
                setattr(group_cfg_item, name, cfg)
            group_cfg_items = {DEFAULT_GROUP_NAME: group_cfg_item}.items()
        else:
            group_cfg_items = cfg_items

        # iterate over all the groups
        for group_name, group_cfg in group_cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # check if the term is a curriculum term
            if not isinstance(group_cfg, RewardGroupCfg):
                raise TypeError(
                    f"Reward group '{group_name}' is not of type 'RewardGroupCfg'. Received: '{type(group_cfg)}'."
                )
            self._group_term_names[group_name] = list()
            self._group_term_cfgs[group_name] = list()
            self._group_class_term_cfgs[group_name] = list()

            # check if config is dict already
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items()

            # iterate over all the terms
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
                # resolve common parameters
                self._resolve_common_term_cfg(f"{group_name}/{term_name}", term_cfg, min_argc=1)
                # add function to list
                self._group_term_names[group_name].append(term_name)
                self._group_term_cfgs[group_name].append(term_cfg)
                # check if the term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_class_term_cfgs[group_name].append(term_cfg)
                    # call reset on the term
                    term_cfg.func.reset()
