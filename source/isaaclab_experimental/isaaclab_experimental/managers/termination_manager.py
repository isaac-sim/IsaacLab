# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination manager for computing done signals for a given world (experimental, Warp-first).

This file mirrors `isaaclab.managers.termination_manager` but switches to a Warp-first,
CUDA-graph-friendly implementation:

- Term functions write into pre-allocated Warp buffers (no per-step torch returns).
- All per-env termination buffers are persistent Warp arrays with torch views at the boundary.
- No data-dependent indexing (e.g. `nonzero`) inside `compute()`; subset updates use masks/kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from prettytable import PrettyTable

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _termination_pre_compute_reset(
    # output
    term_dones: wp.array(dtype=wp.bool, ndim=2),
    truncated: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
    dones: wp.array(dtype=wp.bool),
):
    """Reset per-step termination buffers.

    Launched with dim = (num_envs,) to reset per-env flags and clear the corresponding row in `term_dones`.
    This works even when `term_dones.shape[1] == 0` (no terms).
    """
    env_id = wp.tid()
    truncated[env_id] = False
    terminated[env_id] = False
    dones[env_id] = False
    for term_idx in range(term_dones.shape[1]):
        term_dones[env_id, term_idx] = False


@wp.kernel
def _termination_finalize(
    # input
    term_dones: wp.array(dtype=wp.bool, ndim=2),
    term_is_time_out: wp.array(dtype=wp.bool),
    # output
    truncated: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
    dones: wp.array(dtype=wp.bool),
    last_episode_dones: wp.array(dtype=wp.bool, ndim=2),
):
    """Finalize termination flags and update last-episode term flags (single kernel).

    This kernel:
    - reduces `term_dones` into `truncated`, `terminated`, and `dones`
    - for envs where `dones=True`, copies the current `term_dones` row into `last_episode_dones`
      (matching the stable manager's behavior).
    """
    env_id = wp.tid()

    trunc = bool(False)
    term = bool(False)
    for term_idx in range(term_dones.shape[1]):
        v = term_dones[env_id, term_idx]
        if v:
            if term_is_time_out[term_idx]:
                trunc = True
            else:
                term = True

    done = trunc or term
    truncated[env_id] = trunc
    terminated[env_id] = term
    dones[env_id] = done

    if done:
        for term_idx in range(term_dones.shape[1]):
            last_episode_dones[env_id, term_idx] = term_dones[env_id, term_idx]


# TODO(jichuanh): Look into wp.tile for better performance
@wp.kernel
def _termination_reset_mean_all_2d(
    last_episode_dones: wp.array(dtype=wp.bool, ndim=2),
    term_done_avg: wp.array(dtype=wp.float32),
):
    """Compute mean(done) per term with 2D parallel accumulation."""
    env_id, term_idx = wp.tid()
    num_envs = last_episode_dones.shape[0]
    if num_envs > 0 and last_episode_dones[env_id, term_idx]:
        wp.atomic_add(term_done_avg, term_idx, 1.0 / float(num_envs))


class TerminationManager(ManagerBase):
    """Manager for computing done signals for a given world (Warp-first).

    The termination manager computes the termination signal (also called dones) as a combination
    of termination terms. Each termination term is a function which takes the environment and a
    pre-allocated Warp boolean output buffer and fills it with per-env termination flags.
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._term_cfgs: list[TerminationTermCfg] = list()
        self._class_term_cfgs: list[TerminationTermCfg] = list()

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)

        self._term_name_to_term_idx = {name: i for i, name in enumerate(self._term_names)}

        # persistent buffers (Warp)
        num_terms = len(self._term_names)
        self._term_dones_wp = wp.zeros((self.num_envs, num_terms), dtype=wp.bool, device=self.device)
        self._term_done_avg_wp = wp.zeros((num_terms,), dtype=wp.float32, device=self.device)
        self._last_episode_dones_wp = wp.zeros((self.num_envs, num_terms), dtype=wp.bool, device=self.device)
        self._truncated_wp = wp.zeros((self.num_envs,), dtype=wp.bool, device=self.device)
        self._terminated_wp = wp.zeros((self.num_envs,), dtype=wp.bool, device=self.device)
        self._dones_wp = wp.zeros((self.num_envs,), dtype=wp.bool, device=self.device)

        # per-term flags indicating if a term is a timeout (Warp)
        self._term_is_time_out_wp = wp.array(
            [bool(term_cfg.time_out) for term_cfg in self._term_cfgs], dtype=wp.bool, device=self.device
        )

        # per-term output buffers are views into the columns of `_term_dones_wp` (Warp).
        # This avoids per-term temporary outputs and a per-term "store" kernel.
        # TODO: Investigate performance diff whether it should using row as per env or per term
        self._term_out_views_wp: list[wp.array] = []
        if num_terms > 0:
            row_stride = self._term_dones_wp.strides[0]
            col_stride = self._term_dones_wp.strides[1]
            base_ptr = self._term_dones_wp.ptr
            for term_idx, term_cfg in enumerate(self._term_cfgs):
                out_view = wp.array(
                    ptr=base_ptr + term_idx * col_stride,
                    dtype=wp.bool,
                    shape=(self.num_envs,),
                    strides=(row_stride,),
                    device=self.device,
                )
                self._term_out_views_wp.append(out_view)
                term_cfg.out = out_view

        # torch tensor views (persistent)
        self._term_dones_tensor_view = wp.to_torch(self._term_dones_wp)
        self._last_episode_dones_tensor_view = wp.to_torch(self._last_episode_dones_wp)
        self._truncated_tensor_view = wp.to_torch(self._truncated_wp)
        self._terminated_tensor_view = wp.to_torch(self._terminated_wp)
        self._dones_tensor_view = wp.to_torch(self._dones_wp)
        self._term_done_avg_tensor_view = wp.to_torch(self._term_done_avg_wp)
        self._reset_extras = {
            "Episode_Termination/" + term_name: self._term_done_avg_tensor_view[term_idx]
            for term_idx, term_name in enumerate(self._term_names)
        }

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
        msg += "\n"

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
        """The net termination signal. Shape is (num_envs,)."""
        return self._dones_tensor_view

    @property
    def dones_wp(self) -> wp.array:
        """The net termination signal. Shape is (num_envs,)."""
        return self._dones_wp

    @property
    def time_outs(self) -> torch.Tensor:
        """The timeout signal (reaching max episode length). Shape is (num_envs,)."""
        return self._truncated_tensor_view

    @property
    def time_outs_wp(self) -> wp.array:
        """The timeout signal (reaching max episode length). Shape is (num_envs,)."""
        return self._truncated_wp

    @property
    def terminated(self) -> torch.Tensor:
        """The terminated signal (reaching a terminal state). Shape is (num_envs,)."""
        return self._terminated_tensor_view

    @property
    def terminated_wp(self) -> wp.array:
        """The terminated signal (reaching a terminal state). Shape is (num_envs,)."""
        return self._terminated_wp

    """
    Operations.
    """

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        env_mask: wp.array | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reset termination stats and class terms; return pre-allocated extras.

        Args:
            env_ids: The specific environment indices to reset.
                If None, all environments are considered.
            env_mask: Boolean Warp mask of shape (num_envs,) selecting reset environments.
                If provided, takes precedence over ``env_ids``.

        Returns:
            A dictionary containing the information to log under the "Termination/{term_name}" key.
        """
        # Mask-first path: captured callers must provide env_mask.
        if env_mask is None or not isinstance(env_mask, wp.array):
            if wp.get_device().is_capturing:
                raise RuntimeError(
                    "TerminationManager.reset requires env_mask(wp.array[bool]) during capture. "
                    "Do not pass env_ids on captured paths."
                )
            env_mask = self._env.resolve_env_mask(env_ids=env_ids, env_mask=env_mask)
        if len(self._term_names) > 0:
            self._term_done_avg_wp.zero_()
            wp.launch(
                kernel=_termination_reset_mean_all_2d,
                dim=(self.num_envs, len(self._term_names)),
                inputs=[self._last_episode_dones_wp, self._term_done_avg_wp],
                device=self.device,
            )
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_mask=env_mask)
        return self._reset_extras

    @property
    def episode_termination_extras(self) -> dict[str, torch.Tensor]:
        """Pre-allocated reset logging extras for termination terms."""
        return self._reset_extras

    def compute(self) -> torch.Tensor:
        """Computes the termination signal as union of individual terms.

        Returns:
            The combined termination signal of shape (num_envs,).
        """
        # reset computation (Warp buffers) in a single kernel launch
        wp.launch(
            kernel=_termination_pre_compute_reset,
            dim=self.num_envs,
            inputs=[self._term_dones_wp, self._truncated_wp, self._terminated_wp, self._dones_wp],
            device=self.device,
        )

        # iterate over all the termination terms (fixed list; per-term math is Warp)
        for term_cfg in self._term_cfgs:
            term_cfg.func(self._env, term_cfg.out, **term_cfg.params)

        # finalize dones and update last-episode term flags (single kernel launch)
        wp.launch(
            kernel=_termination_finalize,
            dim=self.num_envs,
            inputs=[
                self._term_dones_wp,
                self._term_is_time_out_wp,
                self._truncated_wp,
                self._terminated_wp,
                self._dones_wp,
                self._last_episode_dones_wp,
            ],
            device=self.device,
        )

        return self._dones_tensor_view

    def get_term(self, name: str) -> torch.Tensor:
        """Returns the termination term value at current step with the specified name.

        Returns:
            The corresponding termination term value. Shape is (num_envs,).
        """
        return self._term_dones_tensor_view[:, self._term_name_to_term_idx[name]]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples for debug/inspection."""
        terms = []
        for i, key in enumerate(self._term_names):
            terms.append((key, [self._term_dones_tensor_view[env_idx, i].float().cpu().item()]))
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
            if not isinstance(term_cfg, TerminationTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type TerminationTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # resolve common parameters (env, out)
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)
