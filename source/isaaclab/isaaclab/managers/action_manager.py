# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action manager for processing actions sent to the environment."""

from __future__ import annotations

import inspect
import logging
import re
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from prettytable import PrettyTable

from ..envs.utils.io_descriptors import GenericActionIODescriptor, _warn_io_descriptors_deprecated
from ..utils import index_fill_, instantiate
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ActionInputCfg, ActionTermCfg

if TYPE_CHECKING:
    from ..assets import AssetBase
    from ..envs import ManagerBasedEnv


logger = logging.getLogger(__name__)


class ActionInput(ManagerTermBase):
    """Select raw input for an existing action controller during interactive playback.

    The action manager owns activated inputs separately from controllers. Input
    construction must not start external resources: the runner retains the input
    before starting its device or session. Inputs never step or reset the scene.
    """

    cfg: ActionInputCfg

    def __init__(self, cfg: ActionInputCfg, env: ManagerBasedEnv, *, action_dim: int) -> None:
        super().__init__(cfg, env)
        self._action_dim = action_dim

    @property
    def action_dim(self) -> int:
        """Dimension of the original controller's raw actions."""
        return self._action_dim

    def prepare_actions(self) -> None:
        """Acquire input once per control step, before any input resolves actions.

        Inputs sharing a device can distribute one sample here regardless of
        controller order. This host-side phase must not advance physics.
        """
        pass

    @abstractmethod
    def resolve_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Select raw actions without modifying the supplied policy tensor.

        The result must have the same shape, device, and dtype as ``actions``.
        Returning ``actions`` unchanged releases control to the policy. The
        manager records the selected actions before any controller processes them.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release input resources before controller or simulation teardown.

        Repeated calls must be harmless. If cleanup fails, retain ownership of
        outstanding resources so a later call can retry safely.
        """
        pass


class ActionTerm(ManagerTermBase):
    """Base class for action terms.

    The action term is responsible for processing the raw actions sent to the environment
    and applying them to the asset managed by the term. The action term is comprised of two
    operations:

    * Processing of actions: This operation is performed once per **environment step** and
      is responsible for pre-processing the raw actions sent to the environment.
    * Applying actions: This operation is performed once per **simulation step** and is
      responsible for applying the processed actions to the asset managed by the term.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        # call the base class constructor
        super().__init__(cfg, env)
        # parse config to obtain asset to which the term is applied
        self._asset: AssetBase = self._env.scene[self.cfg.asset_name]
        self._IO_descriptor = GenericActionIODescriptor()
        self._export_IO_descriptor = True

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        env = getattr(self, "_env", None)
        sim = getattr(env, "sim", None)
        registry = getattr(sim, "vis_marker_registry", None)
        if registry is not None:
            registry.clear_debug_vis_callback(self)

    def close(self) -> None:
        """Release resources owned by the term before simulation teardown.

        Implementations that own external resources should release them here and
        call the parent implementation. Repeated calls must be harmless.
        """
        self._env.sim.vis_marker_registry.clear_debug_vis_callback(self)

    """
    Properties.
    """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action term."""
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        raise NotImplementedError

    @property
    @abstractmethod
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after applying any processing."""
        raise NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the action term has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor for the action term.

        .. deprecated:: 3.0
           IO descriptors will be removed in Isaac Lab 3.2.
        """
        self._IO_descriptor.name = re.sub(r"([a-z])([A-Z])", r"\1_\2", self.__class__.__name__).lower()
        self._IO_descriptor.full_path = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self._IO_descriptor.description = " ".join(self.__class__.__doc__.split())
        self._IO_descriptor.export = self.export_IO_descriptor
        return self._IO_descriptor

    @property
    def export_IO_descriptor(self) -> bool:
        """Whether to export the IO descriptor for the action term.

        .. deprecated:: 3.0
           IO descriptors will be removed in Isaac Lab 3.2.
        """
        return self._export_IO_descriptor

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the action term data.
        Args:
            debug_vis: Whether to visualize the action term data.
        Returns:
            Whether the debug visualization was successfully set. False if the action term does
            not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False

        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                self._debug_vis_handle = self._env.sim.vis_marker_registry.add_debug_vis_callback(self)
        else:
            # remove the subscriber if it exists
            self._env.sim.vis_marker_registry.clear_debug_vis_callback(self)
        # return success
        return True

    @abstractmethod
    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions to process.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """
        raise NotImplementedError

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.
        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.
        This function calls the visualization objects and sets the data to visualize into them.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")


class ActionManager(ManagerBase):
    """Manager for processing and applying actions for a given world.

    The action manager handles the interpretation and application of user-defined
    actions on a given world. It is comprised of different action terms that decide
    the dimension of the expected actions.

    The action manager performs operations at two stages:

    * processing of actions: It splits the input actions to each term and performs any
      pre-processing needed. This should be called once at every environment step.
    * apply actions: This operation typically sets the processed actions into the assets in the
      scene (such as robots). It should be called before every simulation step.
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the action manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
        """
        # check if config is None
        if cfg is None:
            raise ValueError("Action manager configuration is None. Please provide a valid configuration.")

        # call the base class constructor (this prepares the terms)
        self._closed = False
        self._inputs: dict[str, ActionInput] = {}
        try:
            super().__init__(cfg, env)
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.exception("Failed to close action terms after manager initialization failed.")
            raise
        # fixed once the terms are built
        self._action_term_dim = [term.action_dim for term in self._terms.values()]
        self._total_action_dim = sum(self._action_term_dim)
        # create buffers to store actions
        self._action = torch.zeros((self.num_envs, self.total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)

        # check if any term has debug visualization implemented
        self.cfg.debug_vis = False
        for term in self._terms.values():
            self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.action_dim])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def total_action_dim(self) -> int:
        """Total dimension of actions."""
        return self._total_action_dim

    @property
    def active_terms(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return self._action_term_dim

    @property
    def action(self) -> torch.Tensor:
        """The actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        """The previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._prev_action

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    @property
    def get_IO_descriptors(self) -> list[dict[str, Any]]:
        """Get the IO descriptors for the action manager.

        .. deprecated:: 3.0
           IO descriptors will be removed in Isaac Lab 3.2.

        Returns:
            A dictionary with keys as the term names and values as the IO descriptors.
        """
        _warn_io_descriptors_deprecated(stacklevel=3)
        return self._collect_io_descriptors()

    def _collect_io_descriptors(self) -> list[dict[str, Any]]:
        """Collect IO descriptors without emitting a deprecation warning."""
        data = []

        for term_name, term in self._terms.items():
            try:
                data.append(term.IO_descriptor.__dict__.copy())
            except Exception as e:
                print(f"Error getting IO descriptor for term '{term_name}': {e}")

        formatted_data = []
        for item in data:
            name = item.pop("name")
            formatted_item = {"name": name, "extras": item.pop("extras")}
            if not item.pop("export"):
                continue
            for k, v in item.items():
                # Check if v is a tuple and convert to list
                if isinstance(v, tuple):
                    v = list(v)
                if k in ["description", "units"]:
                    formatted_item["extras"][k] = v
                else:
                    formatted_item[k] = v
            formatted_data.append(formatted_item)

        return formatted_data

    """
    Operations.
    """

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        idx = 0
        for name, term in self._terms.items():
            term_actions = self._action[env_idx, idx : idx + term.action_dim].cpu()
            terms.append((name, term_actions.tolist()))
            idx += term.action_dim
        return terms

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the action data.
        Args:
            debug_vis: Whether to visualize the action data.
        Returns:
            Whether the debug visualization was successfully set. False if the action
            does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the action history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """
        # reset the action history
        index_fill_(self._prev_action, env_ids, 0.0)
        index_fill_(self._action, env_ids, 0.0)
        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # Input reset notifications follow every controller's reset.
        for action_input in self._inputs.values():
            action_input.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_action[:] = self._action
        action = action.to(self.device)
        self._action[:] = action

        if not self._inputs:
            # Ordinary training and playback retain the original controller path.
            idx = 0
            for term in self._terms.values():
                term.process_actions(action[:, idx : idx + term.action_dim])
                idx += term.action_dim
            return

        # Acquire shared inputs before resolving any controller's actions.
        for action_input in self._inputs.values():
            action_input.prepare_actions()

        # Resolve all inputs before processing so observations, rewards, and
        # recorders see the same raw actions that the controllers consume.
        idx = 0
        resolved_actions = []
        for name, term in self._terms.items():
            term_actions = action[:, idx : idx + term.action_dim]
            action_input = self._inputs.get(name)
            resolved = term_actions if action_input is None else action_input.resolve_actions(term_actions)
            if action_input is not None:
                if not isinstance(resolved, torch.Tensor):
                    raise TypeError(f"Action input {name!r} must return a torch.Tensor.")
                if resolved.shape != term_actions.shape:
                    raise ValueError(
                        f"Action input {name!r} returned shape {tuple(resolved.shape)};"
                        f" expected {tuple(term_actions.shape)}."
                    )
                if resolved.device != term_actions.device or resolved.dtype != term_actions.dtype:
                    raise ValueError(
                        f"Action input {name!r} must preserve device {term_actions.device}"
                        f" and dtype {term_actions.dtype};"
                        f" received {resolved.device} and {resolved.dtype}."
                    )
                self._action[:, idx : idx + term.action_dim].copy_(resolved)
            resolved_actions.append(resolved)
            idx += term.action_dim

        # Keep controller inputs separate from the manager's recorded history.
        for term, term_actions in zip(self._terms.values(), resolved_actions):
            term.process_actions(term_actions)

    def close(self) -> None:
        """Drain all inputs, then close controllers before simulation teardown."""
        if self._closed:
            return
        self.close_inputs()
        errors = []
        for term in getattr(self, "_terms", {}).values():
            try:
                term.close()
            except Exception as exc:
                logger.error("Failed to close action term %s: %s", type(term).__name__, exc)
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Failed to close {len(errors)} action term(s)") from errors[0]
        self._closed = True

    def activate_input(self, name: str) -> ActionInput:
        """Construct a configured input after the environment is ready.

        Activation leaves the original controller and its configuration intact.
        The caller starts any external device only after this method returns.
        Inputs are owned by this manager until :meth:`close_inputs` succeeds.
        """
        if self._closed:
            raise RuntimeError("Cannot activate input on a closed action manager.")
        term = self._terms[name]
        if name in self._inputs:
            raise RuntimeError(f"Action input {name!r} is already active.")
        if not isinstance(term.cfg.input, ActionInputCfg):
            raise ValueError(f"Action term {name!r} has no ActionInputCfg configured.")
        action_input = instantiate(term.cfg.input, self._env, action_dim=term.action_dim)
        if not isinstance(action_input, ActionInput):
            raise TypeError(f"Configured input for action term {name!r} did not construct an ActionInput.")
        self._inputs[name] = action_input
        return action_input

    def get_input(self, name: str) -> ActionInput | None:
        """Return a term's activated input, or None if none has been activated."""
        if name not in self._terms:
            raise KeyError(name)
        return self._inputs.get(name)

    def close_inputs(self) -> None:
        """Drain activated inputs while their controllers and scene remain alive.

        Every input gets a cleanup attempt. On failure, retain the inputs for a
        retry and leave controller teardown to the caller after a successful drain.
        """
        errors = []
        for name, action_input in self._inputs.items():
            try:
                action_input.close()
            except Exception as exc:
                logger.error("Failed to close action input %s: %s", name, exc)
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Failed to close {len(errors)} action input(s)") from errors[0]
        self._inputs.clear()

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            term.apply_actions()

    def get_term(self, name: str) -> ActionTerm:
        """Returns the action term with the specified name.

        Args:
            name: The name of the action term.

        Returns:
            The action term with the specified name.
        """
        return self._terms[name]

    def serialize(self) -> dict:
        """Serialize the action manager configuration.

        Returns:
            A dictionary of serialized action term configurations.
        """
        return {term_name: term.serialize() for term_name, term in self._terms.items()}

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # create buffers to parse and store terms
        self._term_names: list[str] = []
        self._terms: dict[str, ActionTerm] = {}

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # parse action terms from the config
        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = instantiate(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term
