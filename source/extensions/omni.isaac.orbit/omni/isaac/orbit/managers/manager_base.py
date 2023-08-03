# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import copy
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union

from omni.isaac.orbit.utils import string_to_callable

from .manager_cfg import ManagerBaseTermCfg


class ManagerBase(ABC):
    """Base class for all managers."""

    def __init__(self, cfg: object, env: object):
        """Initialize the manager.

        Args:
            cfg (object): The configuration object.
            env (object): The environment instance.
        """
        # store the inputs
        self.cfg = copy.deepcopy(cfg)
        self._env = env
        # parse config to create terms information
        self._prepare_terms()

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._env.num_envs

    @property
    def device(self) -> str:
        """Device on which to perform computations."""
        return self._env.device

    @property
    @abstractmethod
    def active_terms(self) -> Union[List[str], Dict[str, List[str]]]:
        """Name of active terms."""
        raise NotImplementedError

    """
    Operations.
    """

    def log_info(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, float]:
        """Returns logging information for the current time-step.

        Args:
            env_ids (Sequence[int], optional): The environment ids for which to log data. Defaults
                :obj:`None`, which logs data for all environments.

        Returns:
            Dict[str, float]: Dictionary containing the logging information.
        """
        return {}

    """
    Implementation specific.
    """

    @abstractmethod
    def _prepare_terms(self):
        """Prepare terms information from the configuration object."""
        raise NotImplementedError

    """
    Helper functions.
    """

    def _resolve_common_term_cfg(self, term_name: str, term_cfg: ManagerBaseTermCfg, min_argc: int = 1):
        """Resolve common term configuration.

        Usually, called by the :meth:`_prepare_terms` method to resolve common term configuration.

        Note:
            By default, all term functions are expected to have at least one argument, which is the
            environment object. Some other managers may expect functions to take more arguments, for
            instance, the environment indices as the second argument. In such cases, the
            ``min_argc`` argument can be used to specify the minimum number of arguments
            required by the term function to be called correctly by the manager.

        Args:
            term_name (str): The name of the term.
            term_cfg (ManagerBaseTermCfg): The term configuration.
            min_argc (int): The minimum number of arguments required by the term function to
                be called correctly by the manager.

        Raises:
            TypeError: If the term configuration is not of type :class:`ManagerBaseTermCfg`.
            AttributeError: If the term function is not callable.
            ValueError: If the term function's arguments are not matched by the parameters.
        """
        # check if the term is a valid term config
        if not isinstance(term_cfg, ManagerBaseTermCfg):
            raise TypeError(
                f"Configuration for the term '{term_name}' is not of type ManagerBaseTermCfg. Received '{type(term_cfg)}'."
            )
        # check if a sensor should be enabled
        if term_cfg.sensor_name is not None:
            # TODO: This is a hack. We should not be enabling sensors here.
            #    Instead, we should be enabling sensors in the sensor manager or somewhere else.
            self._env.enable_sensor(term_cfg.sensor_name)
            term_cfg.params["sensor_name"] = term_cfg.sensor_name
        # convert joint names to indices based on regex
        # TODO: What is user wants to penalize joints on one asset and bodies on another?
        if term_cfg.dof_names is not None:
            # check that the asset name is provided
            if term_cfg.asset_name is None:
                raise ValueError(f"The term '{term_name}' requires the asset name to be provided.")
            # acquire the dof indices
            dof_ids, _ = getattr(self._env, term_cfg.asset_name).find_dofs(term_cfg.dof_names)
            term_cfg.params["dof_ids"] = dof_ids
        # convert body names to indices based on regex
        if term_cfg.body_names is not None:
            # check that the asset name is provided
            if term_cfg.asset_name is None:
                raise ValueError(f"The term '{term_name}' requires the asset name to be provided.")
            # acquire the body indices
            body_ids, _ = getattr(self._env, term_cfg.asset_name).find_bodies(term_cfg.body_names)
            term_cfg.params["body_ids"] = body_ids
        # get the corresponding function or functional class
        if isinstance(term_cfg.func, str):
            term_cfg.func = string_to_callable(term_cfg.func)
        # initialize the term if it is a class
        if inspect.isclass(term_cfg.func):
            term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
            # add the "self" argument to the count
            min_argc += 1
        # check if function is callable
        if not callable(term_cfg.func):
            raise AttributeError(f"The term '{term_name}' is not callable. Received: {term_cfg.func}")
        # check if curriculum term's arguments are matched by params
        term_params = list(term_cfg.params.keys())
        args = inspect.getfullargspec(term_cfg.func).args
        # ignore first two arguments for env and env_ids
        # Think: Check for cases when kwargs are set inside the function?
        if len(args) > min_argc:
            if set(args[min_argc:]) != set(term_params):
                msg = f"The term '{term_name}' expects parameters: {args[min_argc:]}, but {term_params} provided."
                raise ValueError(msg)
