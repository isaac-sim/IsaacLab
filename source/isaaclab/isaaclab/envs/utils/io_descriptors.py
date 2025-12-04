# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.assets.articulation import Articulation
    import torch

import dataclasses
import functools
import inspect


@configclass
class GenericActionIODescriptor:
    """Generic action IO descriptor.

    This descriptor is used to describe the action space of a policy.
    It can be extended as needed to add more information about the action term that is being described.
    """

    mdp_type: str = "Action"
    """The type of MDP that the action term belongs to."""

    name: str = None
    """The name of the action term.

    By default, the name of the action term class is used.
    """

    full_path: str = None
    """The full path of the action term class.

    By default, python's will retrieve the path from the file that the action term class is defined in
    and the name of the action term class.
    """

    description: str = None
    """The description of the action term.

    By default, the docstring of the action term class is used.
    """

    shape: tuple[int, ...] = None
    """The shape of the action term.

    This should be populated by the user."""

    dtype: str = None
    """The dtype of the action term.

    This should be populated by the user."""

    action_type: str = None
    """The type of the action term.

    This attribute is purely informative and should be populated by the user."""

    extras: dict[str, Any] = {}
    """Extra information about the action term.

    This attribute is purely informative and should be populated by the user."""

    export: bool = True
    """Whether to export the action term.

    Should be set to False if the class is not meant to be exported.
    """


@configclass
class GenericObservationIODescriptor:
    """Generic observation IO descriptor.

    This descriptor is used to describe the observation space of a policy.
    It can be extended as needed to add more information about the observation term that is being described.
    """

    mdp_type: str = "Observation"
    name: str = None
    full_path: str = None
    description: str = None
    shape: tuple[int, ...] = None
    dtype: str = None
    observation_type: str = None
    extras: dict[str, Any] = {}


# These are defined to help with type hinting
P = ParamSpec("P")
R = TypeVar("R")


# Automatically builds a descriptor from the kwargs
def _make_descriptor(**kwargs: Any) -> GenericObservationIODescriptor:
    """Split *kwargs* into (known dataclass fields) and (extras)."""
    field_names = {f.name for f in dataclasses.fields(GenericObservationIODescriptor)}
    known = {k: v for k, v in kwargs.items() if k in field_names}
    extras = {k: v for k, v in kwargs.items() if k not in field_names}

    desc = GenericObservationIODescriptor(**known)
    # User defined extras are stored in the descriptor under the `extras` field
    desc.extras = extras
    return desc


# Decorator factory for generic IO descriptors.
def generic_io_descriptor(
    _func: Callable[Concatenate[ManagerBasedEnv, P], R] | None = None,
    *,
    on_inspect: Callable[..., Any] | list[Callable[..., Any]] | None = None,
    **descriptor_kwargs: Any,
) -> Callable[[Callable[Concatenate[ManagerBasedEnv, P], R]], Callable[Concatenate[ManagerBasedEnv, P], R]]:
    """
    Decorator factory for generic IO descriptors.

    This decorator can be used in different ways:
    1. The default decorator has all the information I need for my use case:
    ..code-block:: python
        @generic_io_descriptor(GenericIODescriptor(description="..", dtype=".."))
        def my_func(env: ManagerBasedEnv, *args, **kwargs):
            ...
    ..note:: If description is not set, the function's docstring is used to populate it.

    2. I need to add more information to the descriptor:
    ..code-block:: python
        @generic_io_descriptor(description="..", new_var_1="a", new_var_2="b")
        def my_func(env: ManagerBasedEnv, *args, **kwargs):
            ...
    3. I need to add a hook to the descriptor:
    ..code-block:: python
        def record_shape(tensor: torch.Tensor, desc: GenericIODescriptor, **kwargs):
            desc.shape = (tensor.shape[-1],)

        @generic_io_descriptor(description="..", new_var_1="a", new_var_2="b", on_inspect=[record_shape, record_dtype])
        def my_func(env: ManagerBasedEnv, *args, **kwargs):
    ..note:: The hook is called after the function is called, if and only if the `inspect` flag is set when calling the function.

    For example:
    ..code-block:: python
        my_func(env, inspect=True)

    4. I need to add a hook to the descriptor and this hook will write to a variable that is not part of the base descriptor.
    ..code-block:: python
        def record_joint_names(output: torch.Tensor, descriptor: GenericIODescriptor, **kwargs):
            asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
            joint_ids = kwargs["asset_cfg"].joint_ids
            if joint_ids == slice(None, None, None):
                joint_ids = list(range(len(asset.joint_names)))
            descriptor.joint_names = [asset.joint_names[i] for i in joint_ids]

        @generic_io_descriptor(new_var_1="a", new_var_2="b", on_inspect=[record_shape, record_dtype, record_joint_names])
        def my_func(env: ManagerBasedEnv, *args, **kwargs):

    ..note:: The hook can access all the variables in the wrapped function's signature. While it is useful, the user should be careful to
    access only existing variables.

    Args:
        _func: The function to decorate.
        **descriptor_kwargs: Keyword arguments to pass to the descriptor.

    Returns:
        A decorator that can be used to decorate a function.
    """
    # If the decorator is used with a descriptor, use it as the descriptor.
    if _func is not None and isinstance(_func, GenericObservationIODescriptor):
        descriptor = _func
        _func = None
    else:
        descriptor = _make_descriptor(**descriptor_kwargs)

    # Ensures the hook is a list
    if callable(on_inspect):
        inspect_hooks: list[Callable[..., Any]] = [on_inspect]
    else:
        inspect_hooks: list[Callable[..., Any]] = list(on_inspect or [])  # handles None

    def _apply(func: Callable[Concatenate[ManagerBasedEnv, P], R]) -> Callable[Concatenate[ManagerBasedEnv, P], R]:

        # Capture the signature of the function
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(env: ManagerBasedEnv, *args: P.args, **kwargs: P.kwargs) -> R:
            inspect_flag: bool = kwargs.pop("inspect", False)
            out = func(env, *args, **kwargs)
            if inspect_flag:
                # Injects the function's arguments into the hooks and applies the defaults
                bound = sig.bind(env, *args, **kwargs)
                bound.apply_defaults()
                call_kwargs = {
                    "output": out,
                    "descriptor": descriptor,
                    **bound.arguments,
                }
                for hook in inspect_hooks:
                    hook(**call_kwargs)
            return out

        # --- Descriptor bookkeeping ---
        descriptor.name = func.__name__
        descriptor.full_path = f"{func.__module__}.{func.__name__}"
        descriptor.dtype = str(descriptor.dtype)
        # Check if description is set in the descriptor
        if descriptor.description is None:
            descriptor.description = " ".join(func.__doc__.split())

        # Adds the descriptor to the wrapped function as an attribute
        wrapper._descriptor = descriptor
        wrapper._has_descriptor = True
        # Alters the signature of the wrapped function to make it match the original function.
        # This allows the wrapped functions to pass the checks in the managers.
        wrapper.__signature__ = sig
        return wrapper

    # If the decorator is used without parentheses, _func will be the function itself.
    if callable(_func):
        return _apply(_func)
    return _apply


def record_shape(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    """Record the shape of the output tensor.

    Args:
        output: The output tensor.
        descriptor: The descriptor to record the shape to.
        **kwargs: Additional keyword arguments.
    """
    descriptor.shape = (output.shape[-1],)


def record_dtype(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    """Record the dtype of the output tensor.

    Args:
        output: The output tensor.
        descriptor: The descriptor to record the dtype to.
        **kwargs: Additional keyword arguments.
    """
    descriptor.dtype = str(output.dtype)


def record_joint_names(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    """Record the joint names of the output tensor.

    Expects the `asset_cfg` keyword argument to be set.

    Args:
        output: The output tensor.
        descriptor: The descriptor to record the joint names to.
        **kwargs: Additional keyword arguments.
    """
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    joint_ids = kwargs["asset_cfg"].joint_ids
    if joint_ids == slice(None, None, None):
        joint_ids = list(range(len(asset.joint_names)))
    descriptor.joint_names = [asset.joint_names[i] for i in joint_ids]


def record_body_names(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    """Record the body names of the output tensor.

    Expects the `asset_cfg` keyword argument to be set.

    Args:
        output: The output tensor.
        descriptor: The descriptor to record the body names to.
        **kwargs: Additional keyword arguments.
    """
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    body_ids = kwargs["asset_cfg"].body_ids
    if body_ids == slice(None, None, None):
        body_ids = list(range(len(asset.body_names)))
    descriptor.body_names = [asset.body_names[i] for i in body_ids]


def record_joint_pos_offsets(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs):
    """Record the joint position offsets of the output tensor.

    Expects the `asset_cfg` keyword argument to be set.

    Args:
        output: The output tensor.
        descriptor: The descriptor to record the joint position offsets to.
        **kwargs: Additional keyword arguments.
    """
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    ids = kwargs["asset_cfg"].joint_ids
    # Get the offsets of the joints for the first robot in the scene.
    # This assumes that all robots have the same joint offsets.
    descriptor.joint_pos_offsets = asset.data.default_joint_pos[:, ids][0]


def record_joint_vel_offsets(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs):
    """Record the joint velocity offsets of the output tensor.

    Expects the `asset_cfg` keyword argument to be set.

    Args:
        output: The output tensor.
        descriptor: The descriptor to record the joint velocity offsets to.
        **kwargs: Additional keyword arguments.
    """
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    ids = kwargs["asset_cfg"].joint_ids
    # Get the offsets of the joints for the first robot in the scene.
    # This assumes that all robots have the same joint offsets.
    descriptor.joint_vel_offsets = asset.data.default_joint_vel[:, ids][0]


def export_articulations_data(env: ManagerBasedEnv) -> dict[str, dict[str, list[float]]]:
    """Export the articulations data.

    Args:
        env: The environment.

    Returns:
        A dictionary containing the articulations data.
    """
    # Create a dictionary for all the articulations in the scene.
    articulation_joint_data = {}
    for articulation_name, articulation in env.scene.articulations.items():
        # For each articulation, create a dictionary with the articulation's data.
        # Some of the data may be redundant with other information provided by the observation descriptors.
        articulation_joint_data[articulation_name] = {}
        articulation_joint_data[articulation_name]["joint_names"] = articulation.joint_names
        articulation_joint_data[articulation_name]["default_joint_pos"] = (
            articulation.data.default_joint_pos[0].detach().cpu().numpy().tolist()
        )
        articulation_joint_data[articulation_name]["default_joint_vel"] = (
            articulation.data.default_joint_vel[0].detach().cpu().numpy().tolist()
        )
        articulation_joint_data[articulation_name]["default_joint_pos_limits"] = (
            articulation.data.default_joint_pos_limits[0].detach().cpu().numpy().tolist()
        )
        articulation_joint_data[articulation_name]["default_joint_damping"] = (
            articulation.data.default_joint_damping[0].detach().cpu().numpy().tolist()
        )
        articulation_joint_data[articulation_name]["default_joint_stiffness"] = (
            articulation.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
        )
        articulation_joint_data[articulation_name]["default_joint_friction"] = (
            articulation.data.default_joint_friction[0].detach().cpu().numpy().tolist()
        )
        articulation_joint_data[articulation_name]["default_joint_armature"] = (
            articulation.data.default_joint_armature[0].detach().cpu().numpy().tolist()
        )
    return articulation_joint_data


def export_scene_data(env: ManagerBasedEnv) -> dict[str, Any]:
    """Export the scene data.

    Args:
        env: The environment.

    Returns:
        A dictionary containing the scene data.
    """
    # Create a dictionary for the scene data.
    scene_data = {"physics_dt": env.physics_dt, "dt": env.step_dt, "decimation": env.cfg.decimation}
    return scene_data
