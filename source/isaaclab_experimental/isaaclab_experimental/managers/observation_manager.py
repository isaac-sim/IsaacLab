# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation manager for computing observation signals for a given world.

Observations are organized into groups based on their intended usage. This allows having different observation
groups for different types of learning such as asymmetric actor-critic and student-teacher training. Each
group contains observation terms which contain information about the observation function to call, the noise
corruption model to use, and the sensor to retrieve data from.

Each observation group should inherit from the :class:`ObservationGroupCfg` class. Within each group, each
observation term should instantiate the :class:`ObservationTermCfg` class. Based on the configuration, the
observations in a group can be concatenated into a single tensor or returned as a dictionary with keys
corresponding to the term's name.

If the observations in a group are concatenated, the shape of the concatenated tensor is computed based on the
shapes of the individual observation terms. This information is stored in the :attr:`group_obs_dim` dictionary
with keys as the group names and values as the shape of the observation tensor. When the terms in a group are not
concatenated, the attribute stores a list of shapes for each term in the group.

.. note::
    When the observation terms in a group do not have the same shape, the observation terms cannot be
    concatenated. In this case, please set the :attr:`ObservationGroupCfg.concatenate_terms` attribute in the
    group configuration to False.

Observations can also have history. This means a running history is updated per sim step. History can be controlled
per :class:`ObservationTermCfg` (See the :attr:`ObservationTermCfg.history_length` and
:attr:`ObservationTermCfg.flatten_history_dim`). History can also be controlled via :class:`ObservationGroupCfg`
where group configuration overwrites per term configuration if set. History follows an oldest to newest ordering.

The observation manager can be used to compute observations for all the groups or for a specific group. The
observations are computed by calling the registered functions for each term in the group. The functions are
called in the order of the terms in the group. The functions are expected to return a tensor with shape
(num_envs, ...).

If a noise model or custom modifier is registered for a term, the function is called to corrupt
the observation. The corruption function is expected to return a tensor with the same shape as the observation.
The observations are clipped and scaled as per the configuration settings.

Experimental (Warp-first) note:
    Observation term functions follow a Warp-first signature and **write** into pre-allocated Warp buffers:
    ``func(env, out, **params) -> None``. Post-processing may be implemented via Warp kernels where possible.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from prettytable import PrettyTable

from isaaclab.utils import class_to_dict

from isaaclab_experimental.utils import modifiers, noise
from isaaclab_experimental.utils.buffers import CircularBuffer
from isaaclab_experimental.utils.torch_utils import clone_obs_buffer

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ObservationGroupCfg, ObservationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@wp.kernel
def _apply_clip(out: wp.array(dtype=wp.float32, ndim=2), clip_lo: wp.float32, clip_hi: wp.float32):
    env_id = wp.tid()
    for j in range(out.shape[1]):
        out[env_id, j] = wp.clamp(out[env_id, j], clip_lo, clip_hi)


@wp.kernel
def _apply_scale(out: wp.array(dtype=wp.float32, ndim=2), scale: wp.array(dtype=wp.float32)):
    env_id = wp.tid()
    for j in range(out.shape[1]):
        out[env_id, j] = out[env_id, j] * scale[j]


def _resolve_scale_vector(value: Any, dim: int, device: str) -> torch.Tensor:
    """Resolve scale into a (dim,) float32 tensor (defaults to ones)."""
    if value is None:
        return torch.ones((dim,), device=device, dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        t = value.to(device=device, dtype=torch.float32)
        if t.numel() == 1:
            return t.reshape(1).repeat(dim)
        if t.numel() == dim:
            return t.reshape(dim)
        raise ValueError(f"Expected scale tensor with numel=1 or numel={dim}, got {t.numel()}.")
    if isinstance(value, (float, int)):
        return torch.full((dim,), float(value), device=device, dtype=torch.float32)
    if isinstance(value, (tuple, list)):
        if len(value) != dim:
            raise ValueError(f"Expected scale length {dim}, got {len(value)}.")
        return torch.tensor(value, device=device, dtype=torch.float32)
    raise TypeError(f"Unsupported scale type: {type(value)}")


class ObservationManager(ManagerBase):
    """Manager for computing observation signals for a given world.

    Observations are organized into groups based on their intended usage. This allows having different observation
    groups for different types of learning such as asymmetric actor-critic and student-teacher training. Each
    group contains observation terms which contain information about the observation function to call, the noise
    corruption model to use, and the sensor to retrieve data from.

    Each observation group should inherit from the :class:`ObservationGroupCfg` class. Within each group, each
    observation term should instantiate the :class:`ObservationTermCfg` class. Based on the configuration, the
    observations in a group can be concatenated into a single tensor or returned as a dictionary with keys
    corresponding to the term's name.

    If the observations in a group are concatenated, the shape of the concatenated tensor is computed based on the
    shapes of the individual observation terms. This information is stored in the :attr:`group_obs_dim` dictionary
    with keys as the group names and values as the shape of the observation tensor. When the terms in a group are not
    concatenated, the attribute stores a list of shapes for each term in the group.

    .. note::
        When the observation terms in a group do not have the same shape, the observation terms cannot be
        concatenated. In this case, please set the :attr:`ObservationGroupCfg.concatenate_terms` attribute in the
        group configuration to False.

    Observations can also have history. This means a running history is updated per sim step. History can be controlled
    per :class:`ObservationTermCfg` (See the :attr:`ObservationTermCfg.history_length` and
    :attr:`ObservationTermCfg.flatten_history_dim`). History can also be controlled via :class:`ObservationGroupCfg`
    where group configuration overwrites per term configuration if set. History follows an oldest to newest ordering.

    The observation manager can be used to compute observations for all the groups or for a specific group. The
    observations are computed by calling the registered functions for each term in the group. The functions are
    called in the order of the terms in the group. The functions are expected to return a tensor with shape
    (num_envs, ...).

    If a noise model or custom modifier is registered for a term, the function is called to corrupt
    the observation. The corruption function is expected to return a tensor with the same shape as the observation.
    The observations are clipped and scaled as per the configuration settings.

    Experimental (Warp-first) note:
        Observation term functions follow a Warp-first signature and **write** into pre-allocated Warp buffers:
        ``func(env, out, **params) -> None``.
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize observation manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ObservationGroupCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
            RuntimeError: If the shapes of the observation terms in a group are not compatible for concatenation
                and the :attr:`~ObservationGroupCfg.concatenate_terms` attribute is set to True.
        """
        if cfg is None:
            raise ValueError("Observation manager configuration is None. Please provide a valid configuration.")

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)

        # compute combined vector for obs group (matches stable semantics)
        self._group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()
        for group_name, group_term_dims in self._group_obs_term_dim.items():
            # if terms are concatenated, compute the combined shape into a single tuple
            # otherwise, keep the list of shapes as is
            if self._group_obs_concatenate[group_name]:
                try:
                    term_dims = torch.stack([torch.tensor(dims, device="cpu") for dims in group_term_dims], dim=0)
                    if len(term_dims.shape) > 1:
                        if self._group_obs_concatenate_dim[group_name] >= 0:
                            dim = self._group_obs_concatenate_dim[group_name] - 1  # account for the batch offset
                        else:
                            dim = self._group_obs_concatenate_dim[group_name]
                        dim_sum = torch.sum(term_dims[:, dim], dim=0)
                        term_dims[0, dim] = dim_sum
                        term_dims = term_dims[0]
                    else:
                        term_dims = torch.sum(term_dims, dim=0)
                    self._group_obs_dim[group_name] = tuple(term_dims.tolist())
                except RuntimeError:
                    raise RuntimeError(
                        f"Unable to concatenate observation terms in group '{group_name}'."
                        f" The shapes of the terms are: {group_term_dims}."
                        " Please ensure that the shapes are compatible for concatenation."
                        " Otherwise, set 'concatenate_terms' to False in the group configuration."
                    )
            else:
                self._group_obs_dim[group_name] = group_term_dims

        # Stores the latest observations.
        self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None
        # Note: Persistent Warp output buffers (`_group_out_wp` / `_group_out_torch`) and per-term post-processing
        # buffers are allocated during `_prepare_terms()` since they are per-term/per-group setup.

    def __str__(self) -> str:
        """Returns: A string representation for the observation manager."""
        msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"

        # add info for each group
        for group_name, group_dim in self._group_obs_dim.items():
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Observation Terms in Group: '{group_name}'"
            if self._group_obs_concatenate[group_name]:
                table.title += f" (shape: {group_dim})"
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

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []

        if self._obs_buffer is None:
            self.compute()
        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = self._obs_buffer

        for group_name, _ in self._group_obs_dim.items():
            if not self.group_obs_concatenate[group_name]:
                for name, term in obs_buffer[group_name].items():
                    terms.append((group_name + "-" + name, term[env_idx].cpu().tolist()))
                continue

            idx = 0
            # add info for each term
            data = obs_buffer[group_name]
            for name, shape in zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
            ):
                data_length = np.prod(shape)
                term = data[env_idx, idx : idx + data_length]
                terms.append((group_name + "-" + name, term.cpu().tolist()))
                idx += data_length

        return terms

    """
    Properties.
    """

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active observation terms in each group.

        The keys are the group names and the values are the list of observation term names in the group.
        """
        return self._group_obs_term_names

    @property
    def group_obs_dim(self) -> dict[str, tuple[int, ...] | list[tuple[int, ...]]]:
        """Shape of computed observations in each group.

        The key is the group name and the value is the shape of the observation tensor.
        If the terms in the group are concatenated, the value is a single tuple representing the
        shape of the concatenated observation tensor. Otherwise, the value is a list of tuples,
        where each tuple represents the shape of the observation tensor for a term in the group.
        """
        return self._group_obs_dim

    @property
    def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
        """Shape of individual observation terms in each group.

        The key is the group name and the value is a list of tuples representing the shape of the observation terms
        in the group. The order of the tuples corresponds to the order of the terms in the group.
        This matches the order of the terms in the :attr:`active_terms`.
        """
        return self._group_obs_term_dim

    @property
    def group_obs_concatenate(self) -> dict[str, bool]:
        """Whether the observation terms are concatenated in each group or not.

        The key is the group name and the value is a boolean specifying whether the observation terms in the group
        are concatenated into a single tensor. If True, the observations are concatenated along the last dimension.

        The values are set based on the :attr:`~ObservationGroupCfg.concatenate_terms` attribute in the group
        configuration.
        """
        return self._group_obs_concatenate

    @property
    def get_IO_descriptors(self, group_names_to_export: list[str] = ["policy"]):
        """Get the IO descriptors for the observation manager.

        Returns:
            A dictionary with keys as the group names and values as the IO descriptors.
        """
        group_data: dict[str, list[dict[str, Any]]] = {}

        # Collect raw descriptor dicts (plus overloads).
        for group_name in self._group_obs_term_names:
            group_data[group_name] = []
            # check if group name is valid
            if group_name not in self._group_obs_term_names:
                raise ValueError(
                    f"Unable to find the group '{group_name}' in the observation manager."
                    f" Available groups are: {list(self._group_obs_term_names.keys())}"
                )
            for term_name, term_cfg in zip(
                self._group_obs_term_names[group_name], self._group_obs_term_cfgs[group_name]
            ):
                func = term_cfg.func
                if not getattr(func, "_has_descriptor", False):
                    continue
                try:
                    # Both stable-style and Warp-first decorated terms support
                    # the ``inspect=True`` keyword.  Warp-first terms (decorated
                    # with ``generic_io_descriptor_warp``) will NOT execute the
                    # underlying function; their hooks derive metadata from
                    # env/config objects instead.
                    func(self._env, **term_cfg.params, inspect=True)
                    desc = func._descriptor.__dict__.copy()
                    overloads = {}
                    for k in ["modifiers", "clip", "scale", "history_length", "flatten_history_dim"]:
                        if hasattr(term_cfg, k):
                            overloads[k] = getattr(term_cfg, k)
                    desc.update(overloads)
                    group_data[group_name].append(desc)
                except Exception as e:
                    print(f"Error getting IO descriptor for term '{term_name}' in group '{group_name}': {e}")

        formatted_data: dict[str, list[dict[str, Any]]] = {}
        for group_name, data in group_data.items():
            if group_name not in group_names_to_export:
                continue
            formatted_data[group_name] = []
            for item in data:
                name = item.pop("name")
                extras = item.pop("extras", {})
                formatted_item = {"name": name, "overloads": {}, "extras": extras}
                for k, v in item.items():
                    if isinstance(v, tuple):
                        v = list(v)
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy().tolist()
                    if k in ["scale", "clip", "history_length", "flatten_history_dim"]:
                        formatted_item["overloads"][k] = v
                    elif k in ["modifiers", "description", "units"]:
                        formatted_item["extras"][k] = v
                    else:
                        formatted_item[k] = v
                formatted_data[group_name].append(formatted_item)
        return formatted_data

    """
    Operations.
    """

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        env_mask: wp.array | None = None,
    ) -> dict[str, float]:
        # Mask-first path: captured callers must provide env_mask.
        if env_mask is None or not isinstance(env_mask, wp.array):
            # Keep all id->mask resolution strictly outside capture.
            if wp.get_device().is_capturing:
                raise RuntimeError(
                    "ObservationManager.reset requires env_mask(wp.array[bool]) during capture. "
                    "Do not pass env_ids on captured paths."
                )
            env_mask = self._env.resolve_env_mask(env_ids=env_ids, env_mask=env_mask)

        # call all terms that are classes
        for group_name, group_cfg in self._group_obs_class_term_cfgs.items():
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_mask=env_mask)
            # reset terms with history
            for term_name in self._group_obs_term_names[group_name]:
                if term_name in self._group_obs_term_history_buffer[group_name]:
                    self._group_obs_term_history_buffer[group_name][term_name].reset(env_mask=env_mask)
        # call all modifiers/noise models that are classes
        for mod in self._group_obs_class_instances:
            mod.reset(env_mask=env_mask)

        # nothing to log here
        return {}

    def compute(
        self, update_history: bool = False, return_cloned_output: bool = True
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute the observations per group for all groups.

        The method computes the observations for all the groups handled by the observation manager.
        Please check the :meth:`compute_group` on the processing of observations per group.

        Args:
            update_history: The boolean indicator without return obs should be appended to observation history.
                Default to False, in which case calling compute_group does not modify history. This input is no-ops
                if the group's history_length == 0.
            return_cloned_output: Whether to return a cloned snapshot of the observation buffer.
                Set to False to return the persistent internal buffer by reference.

        Returns:
            A dictionary with keys as the group names and values as the computed observations.
            The observations are either concatenated into a single tensor or returned as a dictionary
            with keys corresponding to the term's name.
        """
        # Launch kernels for every group (writes into persistent buffers in-place).
        for group_name in self._group_obs_term_names:
            self.compute_group(group_name, update_history=update_history)
        # Build the obs buffer once (persistent refs to in-place-updated tensors/dicts).
        if self._obs_buffer is None:
            self._obs_buffer = {
                group_name: (
                    self._group_out_torch[group_name]
                    if self._group_use_warp_concat[group_name]
                    else self._group_obs_dict[group_name]
                )
                for group_name in self._group_obs_term_names
            }
        if return_cloned_output:
            return clone_obs_buffer(self._obs_buffer)
        return self._obs_buffer

    def compute_group(self, group_name: str, update_history: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        """Computes the observations for a given group.

        The observations for a given group are computed by calling the registered functions for each
        term in the group. The functions are called in the order of the terms in the group. The functions
        are expected to return a tensor with shape (num_envs, ...).

        The following steps are performed for each observation term:

        1. Compute observation term by calling the function
        2. Apply custom modifiers in the order specified in :attr:`ObservationTermCfg.modifiers`
        3. Apply corruption/noise model based on :attr:`ObservationTermCfg.noise`
        4. Apply clipping based on :attr:`ObservationTermCfg.clip`
        5. Apply scaling based on :attr:`ObservationTermCfg.scale`

        We apply noise to the computed term first to maintain the integrity of how noise affects the data
        as it truly exists in the real world. If the noise is applied after clipping or scaling, the noise
        could be artificially constrained or amplified, which might misrepresent how noise naturally occurs
        in the data.

        Args:
            group_name: The name of the group for which to compute the observations. Defaults to None,
                in which case observations for all the groups are computed and returned.
            update_history: The boolean indicator without return obs should be appended to observation group's history.
                Default to False, in which case calling compute_group does not modify history. This input is no-ops
                if the group's history_length == 0.

        Returns:
            Depending on the group's configuration, the tensors for individual observation terms are
            concatenated along the last dimension into a single tensor. Otherwise, they are returned as
            a dictionary with keys corresponding to the term's name.

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """
        # check if group name is valid
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_obs_term_names[group_name]

        # Persistent per-term obs dict (pre-allocated in _prepare_terms).
        group_obs = self._group_obs_dict[group_name]

        # evaluate terms: compute, add noise, clip, scale, custom modifiers
        for term_name, term_cfg in zip(group_term_names, self._group_obs_term_cfgs[group_name]):
            # compute term's value into pre-allocated Warp output
            term_cfg.func(self._env, term_cfg.out_wp, **term_cfg.params)

            # apply custom modifiers (in-place on out_wp)
            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    modifier.func(term_cfg.out_wp, **modifier.params)

            # apply noise (Warp in-place on out_wp)
            if isinstance(term_cfg.noise, noise.NoiseCfg):
                term_cfg.noise.func(term_cfg.out_wp, term_cfg.noise)
            elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
                term_cfg.noise.func(term_cfg.out_wp)

            # clip then scale (stable semantics); implementation may use Warp kernels
            if term_cfg.clip is not None:
                wp.launch(
                    kernel=_apply_clip,
                    dim=self.num_envs,
                    inputs=[term_cfg.out_wp, float(term_cfg.clip[0]), float(term_cfg.clip[1])],
                    device=self.device,
                )
            if term_cfg.scale is not None:
                wp.launch(
                    kernel=_apply_scale,
                    dim=self.num_envs,
                    inputs=[term_cfg.out_wp, term_cfg.scale_wp],
                    device=self.device,
                )

            # TODO(jichuanh): This is not migrated yet. Need revisit.
            # Update the history buffer if observation term has history enabled
            if term_cfg.history_length > 0:
                # circular buffer is not capture safe
                if wp.get_device().is_capturing:
                    raise RuntimeError(
                        "Observation terms with history (circular buffer) are not CUDA-graph-capture-safe yet. "
                        "Disable history for observation terms used inside a captured graph, or restructure "
                        "the graph to exclude history-buffered terms."
                    )
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                if update_history:
                    circular_buffer.append(wp.to_torch(term_cfg.out_wp))
                elif circular_buffer._buffer is None:
                    # because circular buffer only exits after the simulation steps,
                    # this guards history buffer from corruption by external calls before simulation start
                    circular_buffer = CircularBuffer(
                        max_len=circular_buffer.max_length,
                        batch_size=circular_buffer.batch_size,
                        device=circular_buffer.device,
                    )
                    self._group_obs_term_history_buffer[group_name][term_name] = circular_buffer
                    circular_buffer.append(wp.to_torch(term_cfg.out_wp))

                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
                else:
                    group_obs[term_name] = circular_buffer.buffer

        # return persistent output (updated in-place by kernels above)
        if self._group_use_warp_concat[group_name]:
            return self._group_out_torch[group_name]
        return group_obs

    def serialize(self) -> dict:
        """Serialize the observation term configurations for all active groups.

        Returns:
            A dictionary where each group name maps to its serialized observation term configurations.
        """
        output = {
            group_name: {
                term_name: (
                    term_cfg.func.serialize()
                    if isinstance(term_cfg.func, ManagerTermBase)
                    else {"cfg": class_to_dict(term_cfg)}
                )
                for term_name, term_cfg in zip(
                    self._group_obs_term_names[group_name],
                    self._group_obs_term_cfgs[group_name],
                )
            }
            for group_name in self.active_terms.keys()
        }

        return output

    """
    Helper functions.
    """

    def _prepare_terms(self):  # noqa: C901
        """Prepares a list of observation terms functions."""
        # create buffers to store information for each observation group
        # TODO: Make this more convenient by using data structures.
        self._group_obs_term_names: dict[str, list[str]] = dict()
        self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
        self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_concatenate: dict[str, bool] = dict()
        self._group_obs_concatenate_dim: dict[str, int] = dict()

        self._group_obs_term_history_buffer: dict[str, dict] = dict()
        # create a list to store classes instances, e.g., for modifiers and noise models
        # we store it as a separate list to only call reset on them and prevent unnecessary calls
        self._group_obs_class_instances: list[modifiers.ModifierBase | noise.NoiseModel] = list()

        # Persistent Warp output buffers for concatenated 2D groups (optional fast-path).
        # For other cases (non-concat groups, history outputs, non-2D concat dims), we allocate per-term outputs.
        self._group_out_wp: dict[str, wp.array] = {}
        self._group_out_torch: dict[str, torch.Tensor] = {}
        self._group_use_warp_concat: dict[str, bool] = {}
        self._group_obs_dict: dict[str, dict[str, torch.Tensor]] = {}

        # make sure the simulation is playing since we compute obs dims which needs asset quantities
        if not self._env.sim.is_playing():
            raise RuntimeError(
                "Simulation is not playing. Observation manager requires the simulation to be playing"
                " to compute observation dimensions. Please start the simulation before using the"
                " observation manager."
            )

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
            # group name to list of group term names
            self._group_obs_term_names[group_name] = list()

            self._group_obs_term_dim[group_name] = list()
            self._group_obs_term_cfgs[group_name] = list()
            self._group_obs_class_term_cfgs[group_name] = list()
            group_entry_history_buffer: dict[str, CircularBuffer] = dict()
            # read common config for the group
            self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms
            # to account for the batch dimension
            self._group_obs_concatenate_dim[group_name] = (
                group_cfg.concatenate_dim + 1 if group_cfg.concatenate_dim >= 0 else group_cfg.concatenate_dim
            )
            # check if config is dict already
            if isinstance(group_cfg, dict):
                term_cfg_items = group_cfg.items()
            else:
                term_cfg_items = group_cfg.__dict__.items()
            # iterate over all the terms in each group
            # (we also track raw term dims for Warp output allocation)
            group_term_cfgs: list[ObservationTermCfg] = []
            group_term_raw_dims: list[int] = []
            for term_name, term_cfg in term_cfg_items:
                # skip non-obs settings
                if term_name in [
                    "enable_corruption",
                    "concatenate_terms",
                    "history_length",
                    "flatten_history_dim",
                    "concatenate_dim",
                ]:
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
                # Warp-first signature is (env, out, **params)
                self._resolve_common_term_cfg(f"{group_name}/{term_name}", term_cfg, min_argc=2)

                # check noise settings
                if not group_cfg.enable_corruption:
                    term_cfg.noise = None
                # check group history params and override terms
                if group_cfg.history_length is not None:
                    term_cfg.history_length = group_cfg.history_length
                    term_cfg.flatten_history_dim = group_cfg.flatten_history_dim
                # add term config to list
                self._group_obs_term_names[group_name].append(term_name)
                self._group_obs_term_cfgs[group_name].append(term_cfg)

                # infer dimensions (Warp-first: terms write to out; we infer dim from resolved scene info)
                term_dim = self._infer_term_dim_scalar(term_cfg)
                # Cache the "raw" term output dimension (before history reshaping) for Warp buffer allocation.
                # This matches the tensor shape produced directly by the term into `out`: (num_envs, term_dim).
                term_cfg._term_dim = int(term_dim)
                group_term_cfgs.append(term_cfg)
                group_term_raw_dims.append(int(term_dim))
                obs_dims = (self._env.num_envs, term_dim)

                # if scale is set, check if single float or tuple
                if term_cfg.scale is not None:
                    if not isinstance(term_cfg.scale, (float, int, tuple)):
                        raise TypeError(
                            f"Scale for observation term '{term_name}' in group '{group_name}'"
                            f" is not of type float, int or tuple. Received: '{type(term_cfg.scale)}'."
                        )
                    if isinstance(term_cfg.scale, tuple) and len(term_cfg.scale) != obs_dims[1]:
                        raise ValueError(
                            f"Scale for observation term '{term_name}' in group '{group_name}'"
                            f" does not match the dimensions of the observation. Expected: {obs_dims[1]}"
                            f" but received: {len(term_cfg.scale)}."
                        )

                    scale_vals = (
                        term_cfg.scale if isinstance(term_cfg.scale, tuple) else [float(term_cfg.scale)] * obs_dims[1]
                    )
                    term_cfg.scale_wp = wp.array(scale_vals, dtype=wp.float32, device=self._env.device)

                # prepare modifiers for each observation
                if term_cfg.modifiers is not None:
                    # initialize list of modifiers for term
                    for mod_cfg in term_cfg.modifiers:
                        # check if class modifier and initialize with observation size when adding
                        if isinstance(mod_cfg, modifiers.ModifierCfg):
                            # to list of modifiers
                            if inspect.isclass(mod_cfg.func):
                                if not issubclass(mod_cfg.func, modifiers.ModifierBase):
                                    raise TypeError(
                                        f"Modifier function '{mod_cfg.func}' for observation term '{term_name}'"
                                        f" is not a subclass of 'ModifierBase'. Received: '{type(mod_cfg.func)}'."
                                    )
                                mod_cfg.func = mod_cfg.func(cfg=mod_cfg, data_dim=obs_dims, device=self._env.device)

                                # add to list of class modifiers
                                self._group_obs_class_instances.append(mod_cfg.func)
                        else:
                            raise TypeError(
                                f"Modifier configuration '{mod_cfg}' of observation term '{term_name}' is not of"
                                f" required type ModifierCfg, Received: '{type(mod_cfg)}'"
                            )

                        # check if function is callable
                        if not callable(mod_cfg.func):
                            raise AttributeError(
                                f"Modifier '{mod_cfg}' of observation term '{term_name}' is not callable."
                                f" Received: {mod_cfg.func}"
                            )

                        # check if term's arguments are matched by params
                        term_params = list(mod_cfg.params.keys())
                        args = inspect.signature(mod_cfg.func).parameters
                        args_with_defaults = [arg for arg in args if args[arg].default is not inspect.Parameter.empty]
                        args_without_defaults = [arg for arg in args if args[arg].default is inspect.Parameter.empty]
                        args = args_without_defaults + args_with_defaults
                        # ignore first argument for data
                        if len(args) > 1:
                            if set(args[1:]) != set(term_params + args_with_defaults):
                                raise ValueError(
                                    f"Modifier '{mod_cfg}' of observation term '{term_name}' expects"
                                    f" mandatory parameters: {args_without_defaults[1:]}"
                                    f" and optional parameters: {args_with_defaults}, but received: {term_params}."
                                )

                # prepare noise model classes
                if term_cfg.noise is not None and isinstance(term_cfg.noise, noise.NoiseModelCfg):
                    # plumb the shared per-env RNG state so Warp noise kernels can consume it
                    term_cfg.noise.rng_state_wp = self._env.rng_state_wp
                    noise_model_cls = term_cfg.noise.class_type
                    if not issubclass(noise_model_cls, noise.NoiseModel):
                        raise TypeError(
                            f"Class type for observation term '{term_name}' NoiseModelCfg"
                            f" is not a subclass of 'NoiseModel'. Received: '{type(noise_model_cls)}'."
                        )
                    # initialize func to be the noise model class instance
                    term_cfg.noise.func = noise_model_cls(
                        term_cfg.noise, num_envs=self._env.num_envs, device=self._env.device
                    )
                    self._group_obs_class_instances.append(term_cfg.noise.func)

                # create history buffers and calculate history term dimensions
                if term_cfg.history_length > 0:
                    group_entry_history_buffer[term_name] = CircularBuffer(
                        max_len=term_cfg.history_length, batch_size=self._env.num_envs, device=self._env.device
                    )
                    old_dims = list(obs_dims)
                    old_dims.insert(1, term_cfg.history_length)
                    obs_dims = tuple(old_dims)
                    if term_cfg.flatten_history_dim:
                        obs_dims = (obs_dims[0], np.prod(obs_dims[1:]))
                    raise NotImplementedError("History reshaping is not implemented yet for warp.")

                self._group_obs_term_dim[group_name].append(obs_dims[1:])

                # add term in a separate list if term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_obs_class_term_cfgs[group_name].append(term_cfg)
                    # call reset (in-case internal state should be reset at init)
                    term_cfg.func.reset()

            # Allocate persistent outputs for this group.
            # - If group is concatenated into a flat 2D vector (N, D) with no history terms, allocate a single group
            #   buffer and map term outputs to contiguous slices (fast-path).
            # - Otherwise allocate per-term outputs.
            can_use_group_buffer = (
                self._group_obs_concatenate[group_name]
                and self._group_obs_concatenate_dim[group_name] in (1, -1)
                and all(cfg.history_length == 0 for cfg in group_term_cfgs)
            )

            if can_use_group_buffer:
                total = int(sum(group_term_raw_dims))
                out_wp = wp.zeros((self.num_envs, total), dtype=wp.float32, device=self.device)
                self._group_out_wp[group_name] = out_wp
                self._group_out_torch[group_name] = wp.to_torch(out_wp)

                base_ptr = out_wp.ptr
                row_stride = out_wp.strides[0]
                col_stride = out_wp.strides[1]
                start = 0
                for term_cfg, d in zip(group_term_cfgs, group_term_raw_dims):
                    out_view = wp.array(
                        ptr=base_ptr + start * col_stride,
                        dtype=wp.float32,
                        shape=(self.num_envs, int(d)),
                        strides=(row_stride, col_stride),
                        device=self.device,
                    )
                    term_cfg.out_wp = out_view
                    term_cfg.out_torch = wp.to_torch(term_cfg.out_wp)
                    start += int(d)
            else:
                for term_cfg, d in zip(group_term_cfgs, group_term_raw_dims):
                    term_cfg.out_wp = wp.zeros((self.num_envs, int(d)), dtype=wp.float32, device=self.device)
                    term_cfg.out_torch = wp.to_torch(term_cfg.out_wp)

            # Guard: concat groups must use the Warp fast-path (standard concat dim, no history).
            if self._group_obs_concatenate[group_name] and not can_use_group_buffer:
                raise ValueError(
                    f"Observation group '{group_name}' is concatenated but cannot use the Warp"
                    " fast-path (requires concatenate_dim 0 or -1, and all terms history_length == 0)."
                )

            # Precompute fast-path flag and persistent per-term obs dict.
            self._group_use_warp_concat[group_name] = can_use_group_buffer
            self._group_obs_dict[group_name] = {
                term_name: cfg.out_torch
                for term_name, cfg in zip(self._group_obs_term_names[group_name], group_term_cfgs)
            }

            # add history buffers for each group
            self._group_obs_term_history_buffer[group_name] = group_entry_history_buffer

    def _infer_term_dim_scalar(self, term_cfg: ObservationTermCfg) -> int:
        """Infer observation output dimension (D,) using decorator metadata, scene info, or manager state.

        Resolution order:
        1. ``out_dim`` on the function's ``@generic_io_descriptor_warp`` decorator.
        2. ``axes`` on the decorator (e.g. ``axes=["X","Y","Z"]`` → dim 3).
        3. Explicit ``term_dim`` / ``out_dim`` / ``obs_dim`` in ``term_cfg.params`` (legacy).
        4. ``asset_cfg.joint_ids`` count (joint-based observations).
        """
        # --- 1-2. Decorator metadata (preferred) ---
        func = term_cfg.func
        # Check for descriptor on the (possibly wrapped) function first,
        # then fall back to unwrapping for class-based terms.
        descriptor = getattr(func, "_descriptor", None)
        if descriptor is None and hasattr(func, "__wrapped__"):
            descriptor = getattr(func.__wrapped__, "_descriptor", None)
        if descriptor is not None:
            # 1. Explicit out_dim on decorator
            out_dim = getattr(descriptor, "out_dim", None)
            if out_dim is not None:
                return self._resolve_out_dim(out_dim, term_cfg)
            # 2. Derive from axes metadata
            axes = descriptor.extras.get("axes") if descriptor.extras else None
            if axes is not None:
                return len(axes)

        # --- 3. Legacy explicit override in params ---
        for k in ("term_dim", "out_dim", "obs_dim"):
            if k in term_cfg.params:
                return int(term_cfg.params[k])

        # --- 3. Joint-based fallback via asset_cfg ---
        asset_cfg = term_cfg.params.get("asset_cfg")
        if asset_cfg is None:
            raise ValueError(
                f"Cannot infer output dimension for observation term '{getattr(func, '__name__', func)}'. "
                "Add `out_dim=` to its @generic_io_descriptor_warp decorator."
            )
        asset = self._env.scene[asset_cfg.name]
        joint_ids_wp = getattr(asset_cfg, "joint_ids_wp", None)
        if joint_ids_wp is not None:
            return int(joint_ids_wp.shape[0])
        joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
        if isinstance(joint_ids, slice):
            return int(getattr(asset, "num_joints", wp.to_torch(asset.data.joint_pos).shape[1]))
        return int(len(joint_ids))

    def _resolve_out_dim(self, out_dim: int | str, term_cfg: ObservationTermCfg) -> int:
        """Resolve an ``out_dim`` value from a decorator into a concrete integer.

        Supports:
        - ``int``: returned as-is (fixed dimension).
        - ``"joint"``: number of selected joints from ``asset_cfg``.
        - ``"body:N"``: ``N`` components per selected body from ``asset_cfg``.
        - ``"command"``: query ``command_manager.get_command(name).shape[-1]``.
        - ``"action"``: query ``action_manager.action.shape[-1]``.
        """
        if isinstance(out_dim, int):
            return out_dim

        if out_dim == "joint":
            asset_cfg = term_cfg.params.get("asset_cfg")
            asset = self._env.scene[asset_cfg.name]
            joint_ids_wp = getattr(asset_cfg, "joint_ids_wp", None)
            if joint_ids_wp is not None:
                return int(joint_ids_wp.shape[0])
            joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
            if isinstance(joint_ids, slice):
                return int(getattr(asset, "num_joints", wp.to_torch(asset.data.joint_pos).shape[1]))
            return int(len(joint_ids))

        if isinstance(out_dim, str) and out_dim.startswith("body:"):
            per_body = int(out_dim.split(":")[1])
            asset_cfg = term_cfg.params.get("asset_cfg")
            body_ids = getattr(asset_cfg, "body_ids", None)
            if body_ids is None or body_ids == slice(None):
                asset = self._env.scene[asset_cfg.name]
                return per_body * len(asset.body_names)
            return per_body * len(body_ids)

        if out_dim == "command":
            command_name = term_cfg.params.get("command_name")
            cmd = self._env.command_manager.get_command(command_name)
            return int(cmd.shape[-1])

        if out_dim == "action":
            action = self._env.action_manager.action
            return int(action.shape[-1])

        raise ValueError(f"Unknown out_dim sentinel: {out_dim!r}")
