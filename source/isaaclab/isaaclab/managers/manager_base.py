# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import inspect
import logging
import weakref
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import isaaclab.utils.string as string_utils
from isaaclab.utils import class_to_dict, string_to_callable

from .manager_term_cfg import ManagerTermBaseCfg
from .scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# import logger
logger = logging.getLogger(__name__)


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

    @property
    def __name__(self) -> str:
        """Return the name of the class or subclass."""
        return self.__class__.__name__

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

    def serialize(self) -> dict:
        """General serialization call. Includes the configuration dict."""
        return {"cfg": class_to_dict(self.cfg)}

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
        raise NotImplementedError("The method '__call__' should be implemented by the subclass.")


class ManagerBase(ABC):
    """Base class for all managers."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the manager.

        This function is responsible for parsing the configuration object and creating the terms.

        If the simulation is not playing, the scene entities are not resolved immediately.
        Instead, the resolution is deferred until the simulation starts. This is done to ensure
        that the scene entities are resolved even if the manager is created after the simulation
        has already started.

        Args:
            cfg: The configuration object. If None, the manager is initialized without any terms.
            env: The environment instance.
        """
        # store the inputs
        self.cfg = copy.deepcopy(cfg)
        self._env = env

        # flag for whether the scene entities have been resolved
        # if sim is playing, we resolve the scene entities directly while preparing the terms
        self._is_scene_entities_resolved = self._env.sim.is_playing()

        # if the simulation is not playing, we use callbacks to trigger the resolution of the scene
        # entities configuration. this is needed for cases where the manager is created after the
        # simulation, but before the simulation is playing.
        # FIXME: Once Isaac Sim supports storing this information as USD schema, we can remove this
        #   callback and resolve the scene entities directly inside `_prepare_terms`.
        if not self._env.sim.is_playing():
            # note: Use weakref on all callbacks to ensure that this object can be deleted when its destructor
            # is called
            # The order is set to 20 to allow asset/sensor initialization to complete before the scene entities
            # are resolved. Those have the order 10.
            import omni.timeline  # noqa: PLC0415

            timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
            self._resolve_terms_handle = timeline_event_stream.create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.PLAY),
                lambda event, obj=weakref.proxy(self): obj._resolve_terms_callback(event),
                order=20,
            )
        else:
            self._resolve_terms_handle = None

        # parse config to create terms information
        if self.cfg:
            self._prepare_terms()

    def __del__(self):
        """Delete the manager."""
        if self._resolve_terms_handle:
            self._resolve_terms_handle.unsubscribe()
            self._resolve_terms_handle = None

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

        Please check the :meth:`~isaaclab.utils.string_utils.resolve_matching_names` function for more
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
    Internal callbacks.
    """

    def _resolve_terms_callback(self, event):
        """Resolve configurations of terms once the simulation starts.

        Please check the :meth:`_process_term_cfg_at_play` method for more information.
        """
        # check if scene entities have been resolved
        if self._is_scene_entities_resolved:
            return
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
            # process attributes at runtime
            # these properties are only resolvable once the simulation starts playing
            self._process_term_cfg_at_play(term_name, term_cfg)

        # set the flag
        self._is_scene_entities_resolved = True

    """
    Internal functions.
    """

    def _resolve_common_term_cfg(self, term_name: str, term_cfg: ManagerTermBaseCfg, min_argc: int = 1):
        """Resolve common attributes of the term configuration.

        Usually, called by the :meth:`_prepare_terms` method to resolve common attributes of the term
        configuration. These include:

        * Resolving the term function and checking if it is callable.
        * Checking if the term function's arguments are matched by the parameters.
        * Resolving special attributes of the term configuration like ``asset_cfg``, ``sensor_cfg``, etc.
        * Initializing the term if it is a class.

        The last two steps are only possible once the simulation starts playing.

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

        # get the corresponding function or functional class
        if isinstance(term_cfg.func, str):
            term_cfg.func = string_to_callable(term_cfg.func)
        # check if function is callable
        if not callable(term_cfg.func):
            raise AttributeError(f"The term '{term_name}' is not callable. Received: {term_cfg.func}")

        # check if the term is a class of valid type
        if inspect.isclass(term_cfg.func):
            if not issubclass(term_cfg.func, ManagerTermBase):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ManagerTermBase."
                    f" Received: '{type(term_cfg.func)}'."
                )
            func_static = term_cfg.func.__call__
            min_argc += 1  # forward by 1 to account for 'self' argument
        else:
            func_static = term_cfg.func
        # check if function is callable
        if not callable(func_static):
            raise AttributeError(f"The term '{term_name}' is not callable. Received: {term_cfg.func}")

        # check statically if the term's arguments are matched by params
        term_params = list(term_cfg.params.keys())
        args = inspect.signature(func_static).parameters
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

        # process attributes at runtime
        # these properties are only resolvable once the simulation starts playing
        if self._env.sim.is_playing():
            self._process_term_cfg_at_play(term_name, term_cfg)

    def _process_term_cfg_at_play(self, term_name: str, term_cfg: ManagerTermBaseCfg):
        """Process the term configuration at runtime.

        This function is called when the simulation starts playing. It is used to process the term
        configuration at runtime. This includes:

        * Resolving the scene entity configuration for the term.
        * Initializing the term if it is a class.

        Since the above steps rely on PhysX to parse over the simulation scene, they are deferred
        until the simulation starts playing.

        Args:
            term_name: The name of the term.
            term_cfg: The term configuration.
        """
        for key, value in term_cfg.params.items():
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
                logger.info(msg)
            # store the entity
            term_cfg.params[key] = value

        # initialize the term if it is a class
        if inspect.isclass(term_cfg.func):
            logger.info(f"Initializing term '{term_name}' with class '{term_cfg.func.__name__}'.")
            term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
