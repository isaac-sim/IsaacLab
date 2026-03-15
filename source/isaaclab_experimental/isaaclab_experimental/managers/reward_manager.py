# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward manager for computing reward signals for a given world.

This file is a copy of `isaaclab.managers.reward_manager` placed under
`isaaclab_experimental` so it can evolve independently.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from prettytable import PrettyTable

from isaaclab_experimental.utils.warp.kernels import compute_reset_scale, count_masked

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _sum_and_zero_masked(
    # input
    mask: wp.array(dtype=wp.bool),
    scale: wp.array(dtype=wp.float32),
    # input/output
    episode_sums: wp.array(dtype=wp.float32, ndim=2),
    # output
    out_avg: wp.array(dtype=wp.float32),
):
    term_idx, env_id = wp.tid()
    if mask[env_id]:
        wp.atomic_add(out_avg, term_idx, episode_sums[term_idx, env_id] * scale[0])
        episode_sums[term_idx, env_id] = 0.0


@wp.kernel
def _reward_finalize(
    # input
    term_outs: wp.array(dtype=wp.float32, ndim=2),
    term_weights: wp.array(dtype=wp.float32),
    dt: float,
    # input/output
    reward_buf: wp.array(dtype=wp.float32),
    episode_sums: wp.array(dtype=wp.float32, ndim=2),
    step_reward: wp.array(dtype=wp.float32, ndim=2),
):
    env_id = wp.tid()

    total = wp.float32(0.0)
    for term_idx in range(term_outs.shape[0]):
        weight = term_weights[term_idx]
        if weight != 0.0:
            raw = term_outs[term_idx, env_id]
            weighted = raw * weight
            # store weighted reward rate (matches old: value/dt)
            step_reward[env_id, term_idx] = weighted
            val = weighted * dt
            total += val
            episode_sums[term_idx, env_id] += val

    reward_buf[env_id] = total


@wp.kernel
def _reward_pre_compute_reset(
    # output
    reward_buf: wp.array(dtype=wp.float32),
    step_reward: wp.array(dtype=wp.float32, ndim=2),
    term_outs: wp.array(dtype=wp.float32, ndim=2),
):
    """Reset per-step reward buffers.

    Launched with dim = (num_envs,) to reset `reward_buf` and clear the corresponding row in `step_reward`.
    This works even when `step_reward.shape[1] == 0` (no terms).
    """
    env_id = wp.tid()
    reward_buf[env_id] = 0.0
    for term_idx in range(term_outs.shape[0]):
        step_reward[env_id, term_idx] = 0.0
        term_outs[term_idx, env_id] = 0.0


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
        self._term_name_to_term_idx = {name: i for i, name in enumerate(self._term_names)}

        num_terms = len(self._term_names)
        self._num_terms = num_terms

        # persistent term output buffer (raw, unweighted) laid out as (term, env) for contiguous per-term ops
        self._term_outs_wp = wp.zeros((num_terms, self.num_envs), dtype=wp.float32, device=self.device)
        # per-term output buffers are views into rows of `_term_outs_wp` (Warp)
        self._term_out_views_wp: list[wp.array] = []
        if num_terms > 0:
            row_stride = self._term_outs_wp.strides[0]
            col_stride = self._term_outs_wp.strides[1]
            base_ptr = self._term_outs_wp.ptr
            for term_idx, term_cfg in enumerate(self._term_cfgs):
                out_view = wp.array(
                    ptr=base_ptr + term_idx * row_stride,
                    dtype=wp.float32,
                    shape=(self.num_envs,),
                    strides=(col_stride,),
                    device=self.device,
                )
                self._term_out_views_wp.append(out_view)
                term_cfg.out = out_view

        # prepare extra info to store individual reward term information (warp buffers)
        self._episode_sums_wp = wp.zeros((num_terms, self.num_envs), dtype=wp.float32, device=self.device)
        self._episode_sum_views_wp: dict[str, wp.array] = {}
        if num_terms > 0:
            row_stride = self._episode_sums_wp.strides[0]
            col_stride = self._episode_sums_wp.strides[1]
            base_ptr = self._episode_sums_wp.ptr
            for term_idx, term_name in enumerate(self._term_names):
                sum_view = wp.array(
                    ptr=base_ptr + term_idx * row_stride,
                    dtype=wp.float32,
                    shape=(self.num_envs,),
                    strides=(col_stride,),
                    device=self.device,
                )
                self._episode_sum_views_wp[term_name] = sum_view
        # per-env reward buffer (Warp)
        self._reward_wp = wp.zeros((self.num_envs,), dtype=wp.float32, device=self.device)

        # buffer which stores the current step reward rate for each term for each environment (warp buffer)
        self._step_reward_wp = wp.zeros((self.num_envs, num_terms), dtype=wp.float32, device=self.device)

        # per-term weights stored on-device for single-kernel accumulation
        self._term_weights_wp = wp.array(
            [float(term_cfg.weight) for term_cfg in self._term_cfgs], dtype=wp.float32, device=self.device
        )

        # persistent reset-time logging buffers (warp buffers)
        self._episode_sum_avg_wp = wp.zeros((num_terms,), dtype=wp.float32, device=self.device)
        self._reset_count_wp = wp.zeros((1,), dtype=wp.int32, device=self.device)
        self._reset_scale_wp = wp.zeros((1,), dtype=wp.float32, device=self.device)

        # persistent torch tensor views (helpful for CUDA graph capture)
        self._reward_tensor_view = wp.to_torch(self._reward_wp)
        self._step_reward_tensor_view = wp.to_torch(self._step_reward_wp)
        self._term_weights_tensor_view = wp.to_torch(self._term_weights_wp)
        self._episode_sum_avg_tensor_view = wp.to_torch(self._episode_sum_avg_wp)
        self._reset_extras = {
            "Episode_Reward/" + term_name: self._episode_sum_avg_tensor_view[term_idx]
            for term_idx, term_name in enumerate(self._term_names)
        }

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

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        env_mask: wp.array | None = None,
    ) -> dict[str, torch.Tensor]:
        """Computes/reset episodic reward sums for masked envs (capturable core).

        Args:
            env_ids: The specific environment indices to reset.
                If None, all environments are considered.
            env_mask: Boolean Warp mask of shape (num_envs,) selecting reset environments.
                If provided, takes precedence over ``env_ids``.

        Returns:
            A dictionary containing the information to log under the "Reward/{term_name}" key.
        """
        # Mask-first path: captured callers must provide env_mask.
        if env_mask is None or not isinstance(env_mask, wp.array):
            if wp.get_device().is_capturing:
                raise RuntimeError(
                    "RewardManager.reset requires env_mask(wp.array[bool]) during capture. "
                    "Do not pass env_ids on captured paths."
                )
            env_mask = self._env.resolve_env_mask(env_ids=env_ids, env_mask=env_mask)

        self._episode_sum_avg_wp.zero_()
        self._reset_count_wp.zero_()
        self._reset_scale_wp.zero_()

        wp.launch(kernel=count_masked, dim=self.num_envs, inputs=[env_mask, self._reset_count_wp], device=self.device)
        wp.launch(
            kernel=compute_reset_scale,
            dim=1,
            inputs=[self._reset_count_wp, float(self._env.max_episode_length_s), self._reset_scale_wp],
            device=self.device,
        )

        if self._num_terms > 0:
            wp.launch(
                kernel=_sum_and_zero_masked,
                dim=(self._num_terms, self.num_envs),
                inputs=[env_mask, self._reset_scale_wp, self._episode_sums_wp, self._episode_sum_avg_wp],
                device=self.device,
            )

        # reset all the reward terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_mask=env_mask)

        return self._reset_extras

    def compute(self, dt: float) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # TODO: Investigate performance diff between two .fill_ and kernel launch
        # reset computation (Warp buffers) in a single kernel launch
        wp.launch(
            kernel=_reward_pre_compute_reset,
            dim=self.num_envs,
            inputs=[self._reward_wp, self._step_reward_wp, self._term_outs_wp],
            device=self.device,
        )
        # iterate over all the reward terms (Python loop; per-term math is warp)
        for term_cfg in self._term_cfgs:
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                continue
            # compute term into the persistent warp buffer (raw, unweighted)
            # NOTE: `out` is pre-zeroed every step by `_reward_pre_compute_reset`.
            term_cfg.func(self._env, term_cfg.out, **term_cfg.params)

        # update total reward, episodic sums and step rewards in a single kernel launch
        wp.launch(
            kernel=_reward_finalize,
            dim=self.num_envs,
            inputs=[
                self._term_outs_wp,
                self._term_weights_wp,
                float(dt),
                self._reward_wp,
                self._episode_sums_wp,
                self._step_reward_wp,
            ],
            device=self.device,
        )

        return self._reward_tensor_view

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
        # TODO(jichuanh): it's not guaranteed that the pre-allocated output view is still valid.
        #                 Review this in curriculum manager migration.
        # set the configuration (preserve the pre-allocated output view)
        term_idx = self._term_names.index(term_name)
        cfg.out = self._term_out_views_wp[term_idx]
        self._term_cfgs[term_idx] = cfg
        # keep on-device weights in sync (call this to update weights used in compute)
        self._term_weights_tensor_view[term_idx] = float(cfg.weight)

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
        step_reward_torch = self._step_reward_tensor_view
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
