# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Observation manager for computing observation signals for a given world."""

import torch
from prettytable import PrettyTable
from typing import Dict, List, Tuple

from .manager_base import ManagerBase
from .manager_cfg import ObservationGroupCfg, ObservationTermCfg


class ObservationManager(ManagerBase):
    """Manager for computing observation signals for a given world.

    Observations are organized into groups based on their intended usage. This allows having different observation
    groups for different types of learning such as asymmetric actor-critic and student-teacher training. Each
    group contains observation terms which contain information about the observation function to call, the noise
    corruption model to use, and the sensor to retrieve data from.

    Each observation group should inherit from the :class:`ObservationGroupCfg` class. Within each group, each
    observation term should instantiate the :class:`ObservationTermCfg` class.
    """

    def __init__(self, cfg: object, env: object):
        """Initialize observation manager.

        Args:
            cfg (object): The configuration object or dictionary (``dict[str, ObservationGroupCfg]``).
            env (object): The environment instance.
        """
        super().__init__(cfg, env)
        # compute combined vector for obs group
        self._group_obs_dim: Dict[str, Tuple[int, ...]] = dict()
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
    def active_terms(self) -> Dict[str, List[str]]:
        """Name of active observation terms in each group."""
        return self._group_obs_term_names

    @property
    def group_obs_dim(self) -> Dict[str, Tuple[int, ...]]:
        """Shape of observation tensor in each group."""
        return self._group_obs_dim

    @property
    def group_obs_term_dim(self) -> Dict[str, List[Tuple[int, ...]]]:
        """Shape of observation tensor for each term in each group."""
        return self._group_obs_term_dim

    """
    Operations.
    """

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute the observations per group.

        The method computes the observations for each group and returns a dictionary with keys as
        the group names and values as the computed observations. The observations are computed
        by calling the registered functions for each term in the group. The functions are called
        in the order of the terms in the group. The functions are expected to return a tensor
        with shape ``(num_envs, ...)``. The tensors are then concatenated along the last dimension to
        form the observations for the group.

        If a corruption/noise model is registered for a term, the function is called to corrupt
        the observation. The corruption function is expected to return a tensor with the same
        shape as the observation. The observations are clipped and scaled as per the configuration
        settings. By default, no scaling or clipping is applied.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys as the group names and values as the
                computed observations.
        """
        self._obs_buffer = dict()
        # iterate over all the terms in each group
        for group_name, group_term_names in self._group_obs_term_names.items():
            # buffer to store obs per group
            group_obs = dict.fromkeys(group_term_names, None)
            # read attributes for each term
            obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])
            # evaluate terms: compute, add noise, clip, scale.
            for name, term_cfg in obs_terms:
                # compute term's value
                obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params)
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
                self._obs_buffer[group_name] = torch.cat(list(group_obs.values()), dim=-1)
            else:
                self._obs_buffer[group_name] = group_obs
        # return all group observations
        return self._obs_buffer

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of observation terms functions."""
        # create buffers to store information for each observation group
        # TODO: Make this more convenient by using data structures.
        self._group_obs_term_names: Dict[str, List[str]] = dict()
        self._group_obs_term_dim: Dict[str, List[int]] = dict()
        self._group_obs_term_cfgs: Dict[str, List[ObservationTermCfg]] = dict()
        self._group_obs_concatenate: Dict[str, bool] = dict()

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
                    f"Observation group '{group_name}' is not of type 'ObservationGroupCfg'. Received '{type(group_cfg)}'."
                )
            # initialize list for the group settings
            self._group_obs_term_names[group_name] = list()
            self._group_obs_term_dim[group_name] = list()
            self._group_obs_term_cfgs[group_name] = list()
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
                        f"Configuration for the term '{term_name}' is not of type ObservationTermCfg. Received '{type(term_cfg)}'."
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
