# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward manager for computing reward signals for a given world.

This file is a copy of `isaaclab.managers.reward_manager` placed under
`isaaclab_experimental` so it can evolve independently.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase

from .manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _sum_scaled_selected(
    values: wp.array(dtype=wp.float32), env_ids: wp.array(dtype=wp.int64), out: wp.array(dtype=wp.float32), scale: float
):
    i = wp.tid()
    idx = wp.int32(env_ids[i])
    wp.atomic_add(out, 0, values[idx] * scale)


@wp.kernel
def _zero_selected(values: wp.array(dtype=wp.float32), env_ids: wp.array(dtype=wp.int64)):
    i = wp.tid()
    idx = wp.int32(env_ids[i])
    values[idx] = 0.0


@wp.kernel
def _accumulate_reward_term(
    term_out: wp.array(dtype=wp.float32),
    reward_buf: wp.array(dtype=wp.float32),
    episode_sum: wp.array(dtype=wp.float32),
    step_reward: wp.array(dtype=wp.float32, ndim=2),
    term_idx: int,
    weight: float,
    dt: float,
):
    i = wp.tid()
    raw = term_out[i]
    weighted = raw * weight
    reward_buf[i] += weighted * dt
    episode_sum[i] += weighted * dt
    # store weighted reward rate (matches old: value/dt)
    step_reward[i, term_idx] = weighted


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

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the reward manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RewardTermCfg]``).
            env: The environment instance.
        """

        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._term_cfgs: list[RewardTermCfg] = list()
        self._class_term_cfgs: list[RewardTermCfg] = list()

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        # allocate persistent warp output buffer for each term (raw, unweighted)
        # TODO(jichuanh): What's the best way? Can it be done in the term base class?
        for term_cfg in self._term_cfgs:
            term_cfg.out = wp.zeros((self.num_envs,), dtype=wp.float32, device=self.device)

        # prepare extra info to store individual reward term information (warp buffers)
        self._episode_sums = {}
        for term_name in self._term_names:
            self._episode_sums[term_name] = wp.zeros((self.num_envs,), dtype=wp.float32, device=self.device)
        # create buffer for managing reward per environment (warp buffer)
        self._reward_buf = wp.zeros((self.num_envs,), dtype=wp.float32, device=self.device)

        # buffer which stores the current step reward rate for each term for each environment (warp buffer)
        self._step_reward = wp.zeros((self.num_envs, len(self._term_names)), dtype=wp.float32, device=self.device)

        # persistent "all env ids" buffer for reset() reductions (warp buffer)
        self._all_env_ids_wp = wp.array(list(range(self.num_envs)), dtype=wp.int64, device=self.device)

        # per-term scalar buffers used for reset-time logging (warp buffers)
        self._episode_sum_avg = {}
        for term_name in self._term_names:
            self._episode_sum_avg[term_name] = wp.zeros((1,), dtype=wp.float32, device=self.device)

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
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active reward terms."""
        return self._term_names

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
        extras = {}

        # resolve env_ids into a warp array view (int64 to match torch nonzero dtype)
        if env_ids is None:
            env_ids_wp = self._all_env_ids_wp
            num_ids = self.num_envs
        elif isinstance(env_ids, torch.Tensor):
            env_ids_wp = wp.from_torch(env_ids, dtype=wp.int64)
            num_ids = int(env_ids.numel())
        else:
            env_ids_wp = wp.array(env_ids, dtype=wp.int64, device=self.device)
            num_ids = len(env_ids)

        # compute and reset episodic sums
        for key, episode_sum in self._episode_sums.items():
            avg_scalar = self._episode_sum_avg[key]
            avg_scalar.zero_()
            scale = 1.0 / (num_ids * self._env.max_episode_length_s)
            wp.launch(
                kernel=_sum_scaled_selected,
                dim=num_ids,
                inputs=[episode_sum, env_ids_wp, avg_scalar, scale],
                device=self.device,
            )
            wp.launch(kernel=_zero_selected, dim=num_ids, inputs=[episode_sum, env_ids_wp], device=self.device)

            extras["Episode_Reward/" + key] = wp.to_torch(avg_scalar)
        # reset all the reward terms
        for term_cfg in self._class_term_cfgs:
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
        self._reward_buf.fill_(0.0)
        self._step_reward.fill_(0.0)
        # iterate over all the reward terms (Python loop; per-term math is warp)
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                continue
            # compute term into the persistent warp buffer (raw, unweighted)
            # NOTE: Ensure the term defines all entries each step. This prevents stale values
            # from leaking if a term only conditionally writes to `out`.
            term_cfg.out.fill_(0.0)
            term_cfg.func(self._env, term_cfg.out, **term_cfg.params)
            # update total reward, episodic sums and step rewards in warp
            wp.launch(
                kernel=_accumulate_reward_term,
                dim=self.num_envs,
                inputs=[
                    term_cfg.out,
                    self._reward_buf,
                    self._episode_sums[name],
                    self._step_reward,
                    int(term_idx),
                    float(term_cfg.weight),
                    float(dt),
                ],
                device=self.device,
            )

        return wp.to_torch(self._reward_buf)

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
        if term_name not in self._term_names:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # set the configuration
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> RewardTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the reward term.

        Returns:
            The configuration of the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # return the configuration
        return self._term_cfgs[self._term_names.index(term_name)]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        step_reward_torch = wp.to_torch(self._step_reward)
        for idx, name in enumerate(self._term_names):
            terms.append((name, [step_reward_torch[env_idx, idx].cpu().item()]))
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
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)
