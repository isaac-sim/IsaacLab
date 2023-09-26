# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import carb

from omni.isaac.orbit.utils import string_to_callable

from .manager_cfg import ManagerBaseTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv


class ManagerBase(ABC):
    """Base class for all managers."""

    def __init__(self, cfg: object, env: BaseEnv):
        """Initialize the manager.

        Args:
            cfg: The configuration object.
            env: The environment instance.
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
    def active_terms(self) -> list[str] | dict[str, list[str]]:
        """Name of active terms."""
        raise NotImplementedError

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Resets the manager and returns logging information for the current time-step.

        Args:
            env_ids: The environment ids for which to log data.
                Defaults :obj:`None`, which logs data for all environments.

        Returns:
            Dictionary containing the logging information.
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
            term_name: The name of the term.
            term_cfg: The term configuration.
            min_argc: The minimum number of arguments required by the term function to be called correctly
                by the manager.

        Raises:
            TypeError: If the term configuration is not of type :class:`ManagerBaseTermCfg`.
            ValueError: If the scene entity defined in the term configuration does not exist.
            AttributeError: If the term function is not callable.
            ValueError: If the term function's arguments are not matched by the parameters.
        """
        # check if the term is a valid term config
        if not isinstance(term_cfg, ManagerBaseTermCfg):
            raise TypeError(
                f"Configuration for the term '{term_name}' is not of type ManagerBaseTermCfg. Received '{type(term_cfg)}'."
            )
        # iterate over all the entities and parse the joint and body names
        for key, value in term_cfg.params.items():
            # deal with string
            if isinstance(value, SceneEntityCfg):
                # check if the entity is valid
                if value.name not in self._env.scene.keys():
                    raise ValueError(f"For the term '{term_name}', the scene entity '{value.name}' does not exist.")
                # convert joint names to indices based on regex
                if value.joint_names is not None and value.joint_ids is not None:
                    raise ValueError(
                        f"For the term '{term_name}', both 'joint_names' and 'joint_ids' are specified in '{key}'."
                    )
                if value.joint_names is not None:
                    if isinstance(value.joint_names, str):
                        value.joint_names = [value.joint_names]
                    joint_ids, _ = self._env.scene[value.name].find_joints(value.joint_names)
                    value.joint_ids = joint_ids
                # convert body names to indices based on regex
                if value.body_names is not None and value.body_ids is not None:
                    raise ValueError(
                        f"For the term '{term_name}', both 'body_names' and 'body_ids' are specified in '{key}'."
                    )
                if value.body_names is not None:
                    if isinstance(value.body_names, str):
                        value.body_names = [value.body_names]
                    body_ids, _ = self._env.scene[value.name].find_bodies(value.body_names)
                    value.body_ids = body_ids
                # log the entity for checking later
                msg = f"[{term_cfg.__class__.__name__}:{term_name}] Found entity '{value.name}'."
                if value.joint_ids is not None:
                    msg += f"\n\tJoint names: {value.joint_names} [{value.joint_ids}]"
                if value.body_ids is not None:
                    msg += f"\n\tBody names: {value.body_names} [{value.body_ids}]"
                # print the information
                carb.log_info(msg)
            # store the entity
            term_cfg.params[key] = value

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
        # check if term's arguments are matched by params
        term_params = list(term_cfg.params.keys())
        args = inspect.getfullargspec(term_cfg.func).args
        # ignore first two arguments for env and env_ids
        # Think: Check for cases when kwargs are set inside the function?
        if len(args) > min_argc:
            if set(args[min_argc:]) != set(term_params):
                msg = f"The term '{term_name}' expects parameters: {args[min_argc:]}, but {term_params} provided."
                raise ValueError(msg)
