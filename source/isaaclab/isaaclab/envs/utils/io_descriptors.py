from __future__ import annotations

from isaaclab.utils import configclass
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    import torch

import functools
import inspect
import dataclasses

@configclass
class GenericActionIODescriptor:
    mdp_type: str = "Action"
    name: str = None
    full_path: str = None
    description: str = None
    shape: tuple[int, ...] = None
    dtype: str = None
    action_type: str = None
    extras: dict[str, Any] = {}

@configclass
class GenericIODescriptor:
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
def _make_descriptor(**kwargs: Any) -> GenericIODescriptor:
    """Split *kwargs* into (known dataclass fields) and (extras)."""
    field_names = {f.name for f in dataclasses.fields(GenericIODescriptor)}
    known = {k: v for k, v in kwargs.items() if k in field_names}
    extras = {k: v for k, v in kwargs.items() if k not in field_names}

    desc = GenericIODescriptor(**known)
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

    if _func is not None and isinstance(_func, GenericIODescriptor):
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
            descriptor.description = func.__doc__

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


def record_shape(output: torch.Tensor, descriptor: GenericIODescriptor, **kwargs):
    descriptor.shape = (output.shape[-1],)


def record_dtype(output: torch.Tensor, descriptor: GenericIODescriptor, **kwargs):
    descriptor.dtype = str(output.dtype)


def record_joint_names(output: torch.Tensor, descriptor: GenericIODescriptor, **kwargs):
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    joint_ids = kwargs["asset_cfg"].joint_ids
    if joint_ids == slice(None, None, None):
        joint_ids = list(range(len(asset.joint_names)))
    descriptor.joint_names = [asset.joint_names[i] for i in joint_ids]


def record_body_names(output: torch.Tensor, descriptor: GenericIODescriptor, **kwargs):
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    body_ids = kwargs["asset_cfg"].body_ids
    if body_ids == slice(None, None, None):
        body_ids = list(range(len(asset.body_names)))
    descriptor.body_names = [asset.body_names[i] for i in body_ids]