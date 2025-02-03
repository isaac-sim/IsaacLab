# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.utils import string_to_callable

from .manager_term_cfg import ManagerTermBaseCfg
from .scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ManagerTermBase(ABC):
    """Base class for manager terms.

    Manager term implementations can be functions or classes. If the term is a class, it should
    inherit from this base class and implement the required methods.

    Each manager is implemented as a class that inherits from the :class:`ManagerBase` class. Each manager
    class should also have a corresponding configuration class that defines the configuration terms for the
    manager. Each term should the :class:`ManagerTermBaseCfg` class or its subclass.

    Example pseudo-code for creating a manager:

    .. code-block:: python

        from isaaclab.utils import configclass
        from isaaclab.utils.mdp import ManagerBase, ManagerTermBaseCfg

        @configclass
        class MyManagerCfg:

            my_term_1: ManagerTermBaseCfg = ManagerTermBaseCfg(...)
            my_term_2: ManagerTermBaseCfg = ManagerTermBaseCfg(...)
            my_term_3: ManagerTermBaseCfg = ManagerTermBaseCfg(...)

        # define manager instance
        my_manager = ManagerBase(cfg=ManagerCfg(), env=env)

    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        """Initialize the manager term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        # store the inputs
        self.cfg = cfg
        self._env = env

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

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        pass

    def __call__(self, *args) -> Any:
        """Returns the value of the term required by the manager.

        In case of a class implementation, this function is called by the manager
        to get the value of the term. The arguments passed to this function are
        the ones specified in the term configuration (see :attr:`ManagerTermBaseCfg.params`).

        .. attention::
            To be consistent with memory-less implementation of terms with functions, it is
            recommended to ensure that the returned mutable quantities are cloned before
            returning them. For instance, if the term returns a tensor, it is recommended
            to ensure that the returned tensor is a clone of the original tensor. This prevents
            the manager from storing references to the tensors and altering the original tensors.

        Args:
            *args: Variable length argument list.

        Returns:
            The value of the term.
        """
        raise NotImplementedError


class ManagerBase(ABC):
    """Base class for all managers."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the manager.

        Args:
            cfg: The configuration object. If None, the manager is initialized without any terms.
            env: The environment instance.
        """
        # store the inputs
        self.cfg = copy.deepcopy(cfg)
        self._env = env
        # parse config to create terms information
        if self.cfg:
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
                Defaults None, which logs data for all environments.

        Returns:
            Dictionary containing the logging information.
        """
        return {}

    def find_terms(self, name_keys: str | Sequence[str]) -> list[str]:
        """Find terms in the manager based on the names.

        This function searches the manager for terms based on the names. The names can be
        specified as regular expressions or a list of regular expressions. The search is
        performed on the active terms in the manager.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the term names.

        Returns:
            A list of term names that match the input keys.
        """
        # resolve search keys
        if isinstance(self.active_terms, dict):
            list_of_strings = []
            for names in self.active_terms.values():
                list_of_strings.extend(names)
        else:
            list_of_strings = self.active_terms

        # return the matching names
        return string_utils.resolve_matching_names(name_keys, list_of_strings)[1]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Returns:
            The active terms.
        """
        raise NotImplementedError

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

    def _resolve_common_term_cfg(self, term_name: str, term_cfg: ManagerTermBaseCfg, min_argc: int = 1):
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
            TypeError: If the term configuration is not of type :class:`ManagerTermBaseCfg`.
            ValueError: If the scene entity defined in the term configuration does not exist.
            AttributeError: If the term function is not callable.
            ValueError: If the term function's arguments are not matched by the parameters.
        """
        # check if the term is a valid term config
        if not isinstance(term_cfg, ManagerTermBaseCfg):
            raise TypeError(
                f"Configuration for the term '{term_name}' is not of type ManagerTermBaseCfg."
                f" Received: '{type(term_cfg)}'."
            )
        # iterate over all the entities and parse the joint and body names
        for key, value in term_cfg.params.items():
            # deal with string
            if isinstance(value, SceneEntityCfg):
                # load the entity
                try:
                    value.resolve(self._env.scene)
                except ValueError as e:
                    raise ValueError(f"Error while parsing '{term_name}:{key}'. {e}")
                # log the entity for checking later
                msg = f"[{term_cfg.__class__.__name__}:{term_name}] Found entity '{value.name}'."
                if value.joint_ids is not None:
                    msg += f"\n\tJoint names: {value.joint_names} [{value.joint_ids}]"
                if value.body_ids is not None:
                    msg += f"\n\tBody names: {value.body_names} [{value.body_ids}]"
                # print the information
                omni.log.info(msg)
            # store the entity
            term_cfg.params[key] = value

        # get the corresponding function or functional class
        if isinstance(term_cfg.func, str):
            term_cfg.func = string_to_callable(term_cfg.func)

        # initialize the term if it is a class
        if inspect.isclass(term_cfg.func):
            if not issubclass(term_cfg.func, ManagerTermBase):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ManagerTermBase."
                    f" Received: '{type(term_cfg.func)}'."
                )
            term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
        # check if function is callable
        if not callable(term_cfg.func):
            raise AttributeError(f"The term '{term_name}' is not callable. Received: {term_cfg.func}")

        # check if term's arguments are matched by params
        term_params = list(term_cfg.params.keys())
        args = inspect.signature(term_cfg.func).parameters
        args_with_defaults = [arg for arg in args if args[arg].default is not inspect.Parameter.empty]
        args_without_defaults = [arg for arg in args if args[arg].default is inspect.Parameter.empty]
        args = args_without_defaults + args_with_defaults
        # ignore first two arguments for env and env_ids
        # Think: Check for cases when kwargs are set inside the function?
        if len(args) > min_argc:
            if set(args[min_argc:]) != set(term_params + args_with_defaults):
                raise ValueError(
                    f"The term '{term_name}' expects mandatory parameters: {args_without_defaults[min_argc:]}"
                    f" and optional parameters: {args_with_defaults}, but received: {term_params}."
                )
