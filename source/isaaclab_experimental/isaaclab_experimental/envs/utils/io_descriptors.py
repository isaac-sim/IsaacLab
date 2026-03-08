# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first IO descriptor decorator and inspection hooks (experimental).

This module mirrors the stable :mod:`isaaclab.envs.utils.io_descriptors` but is
designed for Warp-first observation terms whose signature is::

    func(env, out, **params) -> None

Key difference from the stable decorator:
    During inspection (``inspect=True``), the underlying function is **not called**.
    Hooks derive metadata from ``env`` / scene / config objects instead of from a
    returned output tensor.  ``output`` is passed as ``None`` so that hooks share the
    same ``(output, descriptor, **kwargs)`` signature as the stable hooks.

The :class:`GenericObservationIODescriptor` dataclass is reused from the stable
package so that the resulting descriptor dicts are fully compatible with the
existing export / YAML pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

import warp as wp

# Reuse the descriptor dataclass from the stable package.
from isaaclab.envs.utils.io_descriptors import GenericObservationIODescriptor

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs import ManagerBasedEnv

import dataclasses
import functools
import inspect

# These are defined to help with type hinting
P = ParamSpec("P")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


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


# TODO(jichuanh): The exact usage is unclear and this need revisit
# Decorator factory for Warp-first IO descriptors.
def generic_io_descriptor_warp(
    _func: Callable[Concatenate[ManagerBasedEnv, P], R] | None = None,
    *,
    on_inspect: Callable[..., Any] | list[Callable[..., Any]] | None = None,
    **descriptor_kwargs: Any,
) -> Callable[[Callable[Concatenate[ManagerBasedEnv, P], R]], Callable[Concatenate[ManagerBasedEnv, P], R]]:
    """IO descriptor decorator for Warp-first observation terms.

    Works like the stable :func:`generic_io_descriptor` but adapted to the
    ``func(env, out, **params) -> None`` signature:

    * On **normal calls** the decorator passes through to the wrapped function.
    * On **inspection** (``inspect=True`` keyword argument) the wrapped function
      is *not* called.  Instead, the registered hooks are invoked with the same
      ``(output, descriptor, **kwargs)`` contract as the stable hooks, except
      ``output`` is always ``None``.

    This decorator can be used in the same ways as the stable decorator:

    1. With keyword arguments::

        @generic_io_descriptor_warp(observation_type="JointState", units="rad")
        def my_func(env, out, asset_cfg=SceneEntityCfg("robot")) -> None: ...

    2. With a pre-built descriptor::

        @generic_io_descriptor_warp(GenericObservationIODescriptor(description=".."))
        def my_func(env, out, asset_cfg=SceneEntityCfg("robot")) -> None: ...

    3. With inspection hooks::

        @generic_io_descriptor_warp(
            observation_type="JointState",
            on_inspect=[record_joint_names, record_joint_shape, record_joint_pos_offsets],
            units="rad",
        )
        def joint_pos_rel(env, out, asset_cfg=SceneEntityCfg("robot")) -> None: ...

    Args:
        _func: The function to decorate (or a pre-built descriptor).
        on_inspect: Hook(s) called during inspection.
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
            if inspect_flag:
                # Warp-first: do NOT call the function (it requires a pre-allocated
                # ``out`` buffer that does not exist at inspection time).
                # Use bind_partial (tolerates missing ``out``) and apply_defaults so
                # that hooks see resolved default values (e.g. ``asset_cfg``).
                bound = sig.bind_partial(env, **kwargs)
                bound.apply_defaults()
                call_kwargs = {
                    "output": None,
                    "descriptor": descriptor,
                    **bound.arguments,
                }
                for hook in inspect_hooks:
                    hook(**call_kwargs)
                return  # noqa: R502
            return func(env, *args, **kwargs)

        # --- Descriptor bookkeeping ---
        descriptor.name = func.__name__
        descriptor.full_path = f"{func.__module__}.{func.__name__}"
        # Warp-first terms always operate in float32.
        descriptor.dtype = str(descriptor.dtype) if descriptor.dtype is not None else "float32"
        # Check if description is set in the descriptor
        if descriptor.description is None and func.__doc__:
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


# ---------------------------------------------------------------------------
# Inspection hooks
#
# All hooks follow the stable convention: (output, descriptor, **kwargs).
# For Warp-first terms ``output`` is always ``None``; hooks that need shape
# or dtype information must derive it from the scene / config objects in
# **kwargs rather than from the output tensor.
# ---------------------------------------------------------------------------


def record_shape(output: wp.array | None, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    """Record the shape of the output buffer.

    No-op when ``output`` is ``None`` (the typical case during Warp-first
    inspection).  Use a type-specific hook such as :func:`record_joint_shape`
    to derive shape from config instead.

    Args:
        output: The pre-allocated output buffer, or ``None`` during inspection.
        descriptor: The descriptor to record the shape to.
        **kwargs: Additional keyword arguments.
    """
    if output is None:
        return
    descriptor.shape = (output.shape[-1],)


def record_dtype(output: wp.array | None, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    """Record the dtype of the output buffer.

    No-op when ``output`` is ``None`` (the typical case during Warp-first
    inspection — dtype is already set to ``"float32"`` by the decorator).

    Args:
        output: The pre-allocated output buffer, or ``None`` during inspection.
        descriptor: The descriptor to record the dtype to.
        **kwargs: Additional keyword arguments.
    """
    if output is None:
        return
    descriptor.dtype = str(output.dtype)


def record_joint_shape(output: wp.array | None, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    """Derive the observation shape from the resolved ``joint_ids`` count.

    This is the Warp-first alternative to :func:`record_shape` for joint-based
    observations.  It ignores ``output`` and reads the shape from the asset
    configuration instead.

    Args:
        output: Ignored — kept for hook signature compatibility.
        descriptor: The descriptor to update.
        **kwargs: Must contain ``env`` and ``asset_cfg``.
    """
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    joint_ids = kwargs["asset_cfg"].joint_ids
    if joint_ids == slice(None, None, None):
        descriptor.shape = (len(asset.joint_names),)
    else:
        descriptor.shape = (len(joint_ids),)


def record_joint_names(output: wp.array | None, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    """Record the joint names selected by ``asset_cfg.joint_ids``.

    Expects the ``asset_cfg`` keyword argument to be set.

    Args:
        output: Ignored — kept for hook signature compatibility.
        descriptor: The descriptor to record the joint names to.
        **kwargs: Additional keyword arguments.
    """
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    joint_ids = kwargs["asset_cfg"].joint_ids
    if joint_ids == slice(None, None, None):
        joint_ids = list(range(len(asset.joint_names)))
    descriptor.joint_names = [asset.joint_names[i] for i in joint_ids]


def record_body_names(output: wp.array | None, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    """Record the body names selected by ``asset_cfg.body_ids``.

    Expects the ``asset_cfg`` keyword argument to be set.

    Args:
        output: Ignored — kept for hook signature compatibility.
        descriptor: The descriptor to record the body names to.
        **kwargs: Additional keyword arguments.
    """
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    body_ids = kwargs["asset_cfg"].body_ids
    if body_ids == slice(None, None, None):
        body_ids = list(range(len(asset.body_names)))
    descriptor.body_names = [asset.body_names[i] for i in body_ids]


def record_joint_pos_offsets(output: wp.array | None, descriptor: GenericObservationIODescriptor, **kwargs):
    """Record the default joint-position offsets (first env instance).

    Expects the ``asset_cfg`` keyword argument to be set.

    Args:
        output: Ignored — kept for hook signature compatibility.
        descriptor: The descriptor to record the joint position offsets to.
        **kwargs: Additional keyword arguments.
    """
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    ids = kwargs["asset_cfg"].joint_ids
    # Get the offsets of the joints for the first robot in the scene.
    # This assumes that all robots have the same joint offsets.
    descriptor.joint_pos_offsets = wp.to_torch(asset.data.default_joint_pos).clone()[:, ids][0]


def record_joint_vel_offsets(output: wp.array | None, descriptor: GenericObservationIODescriptor, **kwargs):
    """Record the default joint-velocity offsets (first env instance).

    Expects the ``asset_cfg`` keyword argument to be set.

    Args:
        output: Ignored — kept for hook signature compatibility.
        descriptor: The descriptor to record the joint velocity offsets to.
        **kwargs: Additional keyword arguments.
    """
    asset: Articulation = kwargs["env"].scene[kwargs["asset_cfg"].name]
    ids = kwargs["asset_cfg"].joint_ids
    # Get the offsets of the joints for the first robot in the scene.
    # This assumes that all robots have the same joint offsets.
    descriptor.joint_vel_offsets = wp.to_torch(asset.data.default_joint_vel).clone()[:, ids][0]
