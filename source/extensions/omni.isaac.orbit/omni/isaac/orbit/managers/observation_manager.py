# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation manager for computing observation signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ObservationGroupCfg, ObservationTermCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv


class ObservationManager(ManagerBase):
    """Manager for computing observation signals for a given world.

    Observations are organized into groups based on their intended usage. This allows having different observation
    groups for different types of learning such as asymmetric actor-critic and student-teacher training. Each
    group contains observation terms which contain information about the observation function to call, the noise
    corruption model to use, and the sensor to retrieve data from.

    Each observation group should inherit from the :class:`ObservationGroupCfg` class. Within each group, each
    observation term should instantiate the :class:`ObservationTermCfg` class.
    """

    def __init__(self, cfg: object, env: BaseEnv):
        """Initialize observation manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ObservationGroupCfg]``).
            env: The environment instance.
        """
        super().__init__(cfg, env)
        # compute combined vector for obs group
        self._group_obs_dim: dict[str, tuple[int, ...]] = dict()
        for group_name, group_term_dims in self._group_obs_term_dim.items():
            term_dims = [torch.tensor(dims, device="cpu") for dims in group_term_dims]
            self._group_obs_dim[group_name] = tuple(torch.sum(torch.stack(term_dims, dim=0), dim=0).tolist())

    def __str__(self) -> str:
        """Returns: A string representation for the observation manager."""
        msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"

        # add info for each group
        for group_name, group_dim in self._group_obs_dim.items():
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Observation Terms in Group: '{group_name}' (shape: {group_dim})"
            table.field_names = ["Index", "Name", "Shape"]
            # set alignment of table columns
            table.align["Name"] = "l"
            # add info for each term
            obs_terms = zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
            )
            for index, (name, dims) in enumerate(obs_terms):
                # resolve inputs to simplify prints
                tab_dims = tuple(dims)
                # add row
                table.add_row([index, name, tab_dims])
            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active observation terms in each group."""
        return self._group_obs_term_names

    @property
    def group_obs_dim(self) -> dict[str, tuple[int, ...]]:
        """Shape of observation tensor in each group."""
        return self._group_obs_dim

    @property
    def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
        """Shape of observation tensor for each term in each group."""
        return self._group_obs_term_dim

    @property
    def group_obs_concatenate(self) -> dict[str, bool]:
        """Whether the observation terms are concatenated in each group."""
        return self._group_obs_concatenate

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # call all terms that are classes
        for group_cfg in self._group_obs_class_term_cfgs.values():
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    def compute(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute the observations per group for all groups.

        The method computes the observations for all the groups handled by the observation manager.
        Please check the :meth:`compute_group` on the processing of observations per group.

        Returns:
            A dictionary with keys as the group names and values as the computed observations.
        """
        # create a buffer for storing obs from all the groups
        obs_buffer = dict()
        # iterate over all the terms in each group
        for group_name in self._group_obs_term_names:
            obs_buffer[group_name] = self.compute_group(group_name)
        # otherwise return a dict with observations of all groups
        return obs_buffer

    def compute_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
        """Computes the observations for a given group.

        The observations for a given group are computed by calling the registered functions for each
        term in the group. The functions are called in the order of the terms in the group. The functions
        are expected to return a tensor with shape (num_envs, ...).

        If a corruption/noise model is registered for a term, the function is called to corrupt
        the observation. The corruption function is expected to return a tensor with the same
        shape as the observation. The observations are clipped and scaled as per the configuration
        settings.

        The operations are performed in the order: compute, add corruption/noise, clip, scale.
        By default, no scaling or clipping is applied.

        Args:
            group_name: The name of the group for which to compute the observations. Defaults to None,
                in which case observations for all the groups are computed and returned.

        Returns:
            Depending on the group's configuration, the tensors for individual observation terms are
            concatenated along the last dimension into a single tensor. Otherwise, they are returned as
            a dictionary with keys corresponding to the term's name.

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """
        # check ig group name is valid
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_obs_term_names[group_name]
        # buffer to store obs per group
        group_obs = dict.fromkeys(group_term_names, None)
        # read attributes for each term
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])
        # evaluate terms: compute, add noise, clip, scale.
        for name, term_cfg in obs_terms:
            # compute term's value
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
            # apply post-processing
            if term_cfg.noise:
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale:
                obs = obs.mul_(term_cfg.scale)
            # TODO: Introduce delay and filtering models.
            # Ref: https://robosuite.ai/docs/modules/sensors.html#observables
            # add value to list
            group_obs[name] = obs
        # concatenate all observations in the group together
        if self._group_obs_concatenate[group_name]:
            return torch.cat(list(group_obs.values()), dim=-1)
        else:
            return group_obs

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of observation terms functions."""
        # create buffers to store information for each observation group
        # TODO: Make this more convenient by using data structures.
        self._group_obs_term_names: dict[str, list[str]] = dict()
        self._group_obs_term_dim: dict[str, list[int]] = dict()
        self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_concatenate: dict[str, bool] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            group_cfg_items = self.cfg.items()
        else:
            group_cfg_items = self.cfg.__dict__.items()
        # iterate over all the groups
        for group_name, group_cfg in group_cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # check if the term is a curriculum term
            if not isinstance(group_cfg, ObservationGroupCfg):
                raise TypeError(
                    f"Observation group '{group_name}' is not of type 'ObservationGroupCfg'."
                    f" Received: '{type(group_cfg)}'."
                )
            # initialize list for the group settings
            self._group_obs_term_names[group_name] = list()
            self._group_obs_term_dim[group_name] = list()
            self._group_obs_term_cfgs[group_name] = list()
            self._group_obs_class_term_cfgs[group_name] = list()
            # read common config for the group
            self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms

            # check if config is dict already
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items()
            # iterate over all the terms in each group
            for term_name, term_cfg in group_cfg.__dict__.items():
                # skip non-obs settings
                if term_name in ["enable_corruption", "concatenate_terms"]:
                    continue
                # check for non config
                if term_cfg is None:
                    continue
                if not isinstance(term_cfg, ObservationTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type ObservationTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # resolve common terms in the config
                self._resolve_common_term_cfg(f"{group_name}/{term_name}", term_cfg, min_argc=1)
                # check noise settings
                if not group_cfg.enable_corruption:
                    term_cfg.noise = None
                # add term config to list to list
                self._group_obs_term_names[group_name].append(term_name)
                self._group_obs_term_cfgs[group_name].append(term_cfg)
                # call function the first time to fill up dimensions
                obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape[1:])
                self._group_obs_term_dim[group_name].append(obs_dims)
                # add term in a separate list if term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_obs_class_term_cfgs[group_name].append(term_cfg)
                    # call reset (in-case above call to get obs dims changed the state)
                    term_cfg.func.reset()
