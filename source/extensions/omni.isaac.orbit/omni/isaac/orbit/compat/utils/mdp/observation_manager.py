# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Observation manager for computing observation signals for a given world."""

import copy
import functools
import inspect
import torch
from prettytable import PrettyTable
from typing import Any, Callable, Dict, Sequence, Tuple


class ObservationManager:
    """Manager for computing observation signals for a given world.

    Observations are organized into groups based on their intended usage. This allows having different observation
    groups for different types of learning such as asymmetric actor-critic and student-teacher training. Each
    group contains a dictionary of observation terms which contain information about the observation function
    to call, the noise corruption model to use, and the sensor to retrieve data from.

    The configuration for the observation manager is a dictionary of dictionaries. The top level dictionary
    contains a key for each observation group. Each group dictionary contains configuration settings and
    each observation term activated in the group.

    The following configuration settings are available for each group:

    - ``enable_corruption``: A boolean indicating whether to enable noise corruption for the group. If not
      specified, False is assumed.
    - ``return_dict_obs_in_group``: A boolean indicating whether to return the observations as a dictionary
      with the keys being the observation term names, or as a single tensor with the observations concatenated
      tensor along their last dimension. If not specified, False is assumed.


    Each observation term dictionary contains the following keys:

    - ``func``: The name of the observation term function. This is the name of the function to call to compute
      the observation. The function must be defined in the class that inherits from this class. If not specified,
      the function name is assumed to be the same as the term name.
    - ``scale``: The scale to apply to the observation before adding noise. If not specified, no scaling is
      applied.
    - ``clip``: The clipping range to apply to the observation after adding noise. This is a tuple of the
      form (min, max). If not specified, no clipping is applied.
    - ``noise``: The noise corruption model to use. If not specified, no corruption is done. This is a dictionary
      with the following keys:

      - ``type``: The type of noise corruption model to use. This can be either:

        - ``deterministic``: Deterministic additive noise.
        - ``gaussian``: Gaussian noise is added.
        - ``uniform``: Uniform noise is added.

      - ``params``: The parameters for the noise corruption model. This is a dictionary with the keys
        depending on the type of noise corruption model:

        - For deterministic noise:

          - ``value``: The value to add to the observation.

        - For gaussian noise:

          - ``mean``: The mean of the Gaussian distribution.
          - ``std``: The standard deviation of the Gaussian distribution.

        - For uniform noise:

          - ``min``: The minimum value of the uniform distribution.
          - ``max``: The maximum value of the uniform distribution.

    Usage:
        .. code-block:: python

            import torch
            from collections import namedtuple

            from omni.isaac.orbit.utils.mdp.observation_manager import ObservationManager


            class DummyObservationManager(ObservationManager):
                def obs_term_1(self, env):
                    return torch.ones(env.num_envs, 4, device=self.device)

                def obs_term_2(self, env, bbq: bool):
                    return bbq * torch.ones(env.num_envs, 1, device=self.device)

                def obs_term_3(self, env, hot: bool):
                    return hot * 2 * torch.ones(env.num_envs, 1, device=self.device)

                def obs_term_4(self, env, hot: bool, bland: float):
                    return hot * bland * torch.ones(env.num_envs, 5, device=self.device)


            # dummy environment with 20 instances
            env = namedtuple("IsaacEnv", ["num_envs"])(20)
            # dummy device
            device = "cpu"

            # create observations manager
            cfg = {
                "return_dict_obs_in_group": False,
                "policy": {
                    "enable_corruption": True,
                    "obs_term_1": {
                        "scale": 10,
                        "noise": {"name": "uniform", "min": -0.1, "max": 0.1}
                    },
                },
                "critic": {
                    "enable_corruption": False,
                    "obs_term_1": {"scale": 10},
                    "obs_term_2": {"scale": 5, "bbq": True},
                    "obs_term_3": {"scale": 0.0, "hot": False}
                },
            }
            obs_man = DummyObservationManager(cfg, env, device)

            # print observation manager
            # shows sequence of observation terms for each group and their configuration
            print(obs_man)

            # here we reset all environment ids, but it is possible to reset only a subset
            # of the environment ids based on the episode termination.
            obs_man.reset_idx(env_ids=list(range(env.num_envs)))

            # check number of active terms
            assert len(obs_man.active_terms["policy"]) == 1
            assert len(obs_man.active_terms["critic"]) == 3

            # compute observation using manager
            observations = obs_man.compute()
            # check the observation shape
            assert (env.num_envs, 4) == observations["policy"].shape
            assert (env.num_envs, 6) == observations["critic"].shape

    """

    def __init__(self, cfg: Dict[str, Dict[str, Any]], env, device: str):
        """Initialize observation manager.

        Args:
            cfg (Dict[str, Dict[str, Any]]): A dictionary of configuration settings for each group.
            env (_type_): The object instance (typically the environment) from which data is read to
                compute the observation terms.
            device (str): The computation device for observations.
        """
        # store input
        self._cfg = copy.deepcopy(cfg)
        self._env = env
        self._device = device
        # store observation manager settings
        self._return_dict_obs_in_group = self._cfg.get("return_dict_obs_in_group", False)
        # parse config to create observation terms information
        self._prepare_observation_terms()
        # compute combined vector for obs group
        self._group_obs_dim: Dict[str, Tuple[int, ...]] = dict()
        for group_name, group_term_dims in self._group_obs_term_dim.items():
            term_dims = [torch.tensor(dims, device="cpu") for dims in group_term_dims]
            self._group_obs_dim[group_name] = tuple(torch.sum(torch.stack(term_dims, dim=0), dim=0).tolist())

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"

        # add info for each group
        for group_name, group_dim in self._group_obs_dim.items():
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Observation Terms in Group: '{group_name}' (shape: {group_dim})"
            table.field_names = ["Index", "Name", "Function", "Shape", "Corruptor", "Clipping", "Scaling", "Parameters"]
            # set alignment of table columns
            table.align["Name"] = "l"
            # add info for each term
            obs_terms = zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_functions[group_name],
                self._group_obs_term_dim[group_name],
                self._group_obs_term_corruptors[group_name],
                self._group_obs_term_clip_ranges[group_name],
                self._group_obs_term_scales[group_name],
                self._group_obs_term_params[group_name],
            )
            for index, (name, func, dims, corruptor, clip_range, scale, params) in enumerate(obs_terms):
                # resolve inputs to simplify prints
                tab_func = func.__name__
                tab_dims = tuple(dims)
                tab_corruptor = corruptor.__str__()
                tab_scale = 1.0 if scale is None else scale
                tab_params = None if not any(params) else params
                # add row
                table.add_row([index, name, tab_func, tab_dims, tab_corruptor, clip_range, tab_scale, tab_params])
            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def device(self) -> str:
        """Name of device for computation."""
        return self._device

    @property
    def active_terms(self) -> Dict[str, Sequence[str]]:
        """Name of active observation terms in each group."""
        return self._group_obs_term_names

    @property
    def group_obs_dim(self) -> Dict[str, Tuple[int, ...]]:
        """Shape of observation tensor in each group."""
        return self._group_obs_dim

    """
    Operations.
    """

    def reset_idx(self, env_ids: torch.Tensor):
        """Reset the terms computation for input environment indices.

        Args:
            env_ids (torch.Tensor): Indices of environment instances to reset.
        """
        # Might need this when dealing with history and delays.
        pass

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
            obs_terms = zip(
                group_term_names,
                self._group_obs_term_corruptors[group_name],
                self._group_obs_term_clip_ranges[group_name],
                self._group_obs_term_scales[group_name],
                self._group_obs_term_params[group_name],
                self._group_obs_term_functions[group_name],
            )
            # evaluate terms: compute, add noise, clip, scale.
            for name, corruptor, clip_range, scale, params, func in obs_terms:
                # compute term's value
                obs: torch.Tensor = func(self._env, **params)
                # apply post-processing
                if corruptor:
                    obs = corruptor(obs)
                if clip_range:
                    obs = obs.clip_(min=clip_range[0], max=clip_range[1])
                if scale:
                    obs = obs.mul_(scale)
                # TODO: Introduce delay and filtering models.
                # Ref: https://robosuite.ai/docs/modules/sensors.html#observables
                # add value to list
                group_obs[name] = obs
            # concatenate all observations in the group together
            if self._return_dict_obs_in_group:
                self._obs_buffer[group_name] = group_obs
            else:
                self._obs_buffer[group_name] = torch.cat(list(group_obs.values()), dim=-1)
        # return all group observations
        return self._obs_buffer

    """
    Noise models.
    """

    @staticmethod
    def _add_deterministic_noise(obs: torch.Tensor, value: float):
        """Add a constant noise."""
        return obs + value

    @staticmethod
    def _add_uniform_noise(obs: torch.Tensor, min: float, max: float):
        """Adds a noise sampled from a uniform distribution (-min_noise, max_noise)."""
        return obs + torch.rand_like(obs) * (max - min) + min

    @staticmethod
    def _add_gaussian_noise(obs: torch.Tensor, mean: float, std: float):
        return obs + mean + std * torch.randn_like(obs)

    """
    Helper functions.
    """

    def _prepare_observation_terms(self):
        """Prepares a list of observation terms functions.

        Raises:
            AttributeError: If the observation term's function not found in the class.
        """
        # create buffers to store information for each observation group
        # TODO: Make this more convenient by using data structures.
        self._group_obs_term_names: Dict[str, Sequence[str]] = dict()
        self._group_obs_term_dim: Dict[str, Sequence[int]] = dict()
        self._group_obs_term_params: Dict[str, Sequence[Dict[str, float]]] = dict()
        self._group_obs_term_clip_ranges: Dict[str, Sequence[Tuple[float, float]]] = dict()
        self._group_obs_term_scales: Dict[str, Sequence[float]] = dict()
        self._group_obs_term_corruptors: Dict[str, Sequence[Callable]] = dict()
        self._group_obs_term_functions: Dict[str, Sequence[Callable]] = dict()

        for group_name, group_cfg in self._cfg.items():
            # skip non-group settings (those should be read above)
            if not isinstance(group_cfg, dict):
                continue
            # initialize list for the group settings
            self._group_obs_term_names[group_name] = list()
            self._group_obs_term_dim[group_name] = list()
            self._group_obs_term_params[group_name] = list()
            self._group_obs_term_clip_ranges[group_name] = list()
            self._group_obs_term_scales[group_name] = list()
            self._group_obs_term_corruptors[group_name] = list()
            self._group_obs_term_functions[group_name] = list()
            # read common config for the group
            enable_corruption = group_cfg.pop("enable_corruption", False)
            # parse group observation settings
            for term_name, term_cfg in group_cfg.items():
                # skip non-obs settings (those should be read above)
                if not isinstance(term_cfg, dict):
                    continue
                # read term's function
                # if not specified, assume the function name is the same as the term name
                term_func = term_cfg.get("func", term_name)
                # check if obs manager has the term
                if hasattr(self, term_func):
                    # check noise settings
                    noise_cfg = term_cfg.pop("noise", None)
                    if enable_corruption and noise_cfg:
                        # read noise parameters
                        noise_name = noise_cfg.pop("name")
                        noise_params = noise_cfg
                        # create a wrapper s.t. Callable[[torch.Tensor], torch.Tensor]
                        noise_func = getattr(self, f"_add_{noise_name}_noise")
                        noise_func = functools.partial(noise_func, **noise_params)
                        # make string representation nicer
                        noise_func.__str__ = lambda: f"Noise: {noise_name}, Params: {noise_params}"
                        # add function to list
                        self._group_obs_term_corruptors[group_name].append(noise_func)
                    else:
                        self._group_obs_term_corruptors[group_name].append(None)
                    # check clip_range and scale settings
                    self._group_obs_term_clip_ranges[group_name].append(term_cfg.pop("clip", None))
                    self._group_obs_term_scales[group_name].append(term_cfg.pop("scale", None))
                    # check function for observation term
                    func = getattr(self, term_name)
                    # check if term's arguments are matched by params
                    term_params = list(term_cfg.keys())
                    args = inspect.getfullargspec(func).args
                    # ignore first two arguments for (self, env)
                    # Think: Check for cases when kwargs are set inside the function?
                    if len(args) > 2:
                        if set(args[2:]) != set(term_params):
                            msg = f"Observation term '{term_name}' expects parameters: {args[2:]}, but {term_params} provided."
                            raise ValueError(msg)
                    # add function to list
                    self._group_obs_term_names[group_name].append(term_name)
                    self._group_obs_term_functions[group_name].append(func)
                    self._group_obs_term_params[group_name].append(term_cfg)
                    # call function the first time to fill up dimensions
                    obs_dims = tuple(func(self._env, **term_cfg).shape[1:])
                    self._group_obs_term_dim[group_name].append(obs_dims)
                else:
                    raise AttributeError(f"Observation term with the name '{term_name}' has no implementation.")
