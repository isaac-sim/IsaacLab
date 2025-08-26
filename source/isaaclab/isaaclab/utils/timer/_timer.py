# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for a timer class that can be used for performance measurements."""

from __future__ import annotations

import functools
import inspect
import sys
import weakref
from collections.abc import Callable
from typing import Dict, ParamSpec, Tuple, Type, TypeVar, cast

from isaaclab.utils.timer._core import Timer

# -- Global registries ------------------------------------------------------------------------------------------------

# Free-functions per group: {group: {(module, name): (raw, deco)}}
_func_toggle_registry: dict[str, dict[tuple[str, str], tuple[Callable, Callable]]] = {}

# Dynamic (local-scope) functions per group: {group: [ {"target": target_list, "raw": raw, "deco": deco} ]}
_dynamic_func_registry = {}  # {group: [ {"target": target_list, "raw": raw, "deco": deco} ]}

# Classes per group: {group: WeakSet({Class, ...})}
_class_group_registry: dict[str, weakref.WeakSet[type]] = {}

# -- metaclass: registers classes that have timed methods -------------------------------------------------------------


class TimerToggleMeta(type):
    """
    A metaclass that allows to instrument a class with a timer.
    """

    def __init__(cls, name, bases, ns):
        super().__init__(name, bases, ns)
        reg = getattr(cls, "_timer_toggle_registry", None)
        if not reg:
            return
        for group in reg.keys():
            _class_group_registry.setdefault(group, weakref.WeakSet()).add(cls)


class Instrumented(metaclass=TimerToggleMeta):
    """
    A metaclass that allows to instrument a class with a timer.
    """

    pass


# -- decorators ------------------------------------------------------------------------------------------------------

# Mock up a parameter specification and return type
P = ParamSpec("P")
R = TypeVar("R")


def _wrap_with_timer(fn: Callable[P, R], group: str, timer_kwargs: dict) -> Callable[P, R]:
    """Wrap a function with a timer.

    Checks if the function is a coroutine (async function) and wraps it accordingly by waiting for the coroutine to complete.

    Args:
        fn: The function to wrap.
        group: The group to time.
        timer_kwargs: The keyword arguments to pass to the timer.

    Returns:
        The wrapped function.
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapped_async(*a: P.args, **k: P.kwargs) -> R:
            with Timer(group=group, **timer_kwargs):
                return await fn(*a, **k)

        return wrapped_async
    else:

        @functools.wraps(fn)
        def wrapped(*a: P.args, **k: P.kwargs) -> R:
            with Timer(group=group, **timer_kwargs):
                return fn(*a, **k)

        return wrapped


def _looks_like_method(fn: Callable) -> bool:
    """Check if a function is a method.

    Args:
        fn: The function to check.

    Returns:
        True if the function is a method, False otherwise.
    """
    # "ClassName.func" vs "func" (top-level) or "<locals>"
    parts = getattr(fn, "__qualname__", "").split(".")
    return len(parts) >= 2 and parts[-2] != "<locals>"


def timer(group: str = "default", **timer_kwargs):
    """Decorator to time a function or a method.

    It has different behavior for class methods and free functions:
     - Class methods: installs a descriptor; methods are swapped on the class (zero steady-state overhead after toggle).
     - Free functions: registers raw/decorated pair; we rebind the module global on toggle (zero steady-state overhead
      when called via module).

    Usage:


    Notes:
        - Put @timer(...) **closest to def** when stacking with @staticmethod/@classmethod.
        - Calls like 'from mod import func' hold their own reference and won't
          see future rebinding; prefer 'import mod; mod.func(...)' in hot paths.
    """

    class _ToggleDescriptor:
        """A descriptor that wraps a function with a timer.

        Should not be called directly. Do not use metaclass/descriptor for inline.
        Use the `timer` decorator instead to add instrumenting behavior.
        """

        __slots__ = ("fn", "decorated", "name")

        def __init__(self, fn: Callable[P, R]) -> None:
            """Initialize the descriptor.

            Args:
                fn: The function to wrap.
                group: The group to time.
                timer_kwargs: The keyword arguments to pass to the timer.
            """
            self.fn = fn
            self.decorated = _wrap_with_timer(fn, group, timer_kwargs)
            self.name = None

        def __set_name__(self, owner, name) -> None:
            """Set the name of the descriptor.

            Args:
                owner: The class that the descriptor is being set on.
                name: The name of the descriptor.
            """
            self.name = name
            # Per-class registry
            reg = getattr(owner, "_timer_toggle_registry", {})
            reg.setdefault(group, {})[name] = (self.fn, self.decorated)
            setattr(owner, "_timer_toggle_registry", reg)
            # Track this class for global toggles
            _class_group_registry.setdefault(group, weakref.WeakSet()).add(owner)
            # Install chosen implementation (no branch at callsite)
            setattr(owner, name, self.decorated if Timer.group_enable.get(group, True) else self.fn)

        def __call__(self, *a: P.args, **k: P.kwargs) -> R:  # never actually called at runtime
            """Should not be called directly. Do not use metaclass/descriptor for inline."""
            raise RuntimeError("Descriptor should not be called directly")

    def apply(fn: Callable[P, R]) -> Callable[P, R]:
        """Apply the timer decorator to a function.

        Args:
            fn: The function to wrap.

        Returns:
            The wrapped function.
        """
        if _looks_like_method(fn):
            # Inside a class body: return descriptor so __set_name__ runs.
            return cast(Callable[P, R], _ToggleDescriptor(fn))
        # Free function: register and return currently selected impl.
        raw = fn
        deco = _wrap_with_timer(fn, group, timer_kwargs)
        _func_toggle_registry.setdefault(group, {})[(fn.__module__, fn.__name__)] = (raw, deco)
        return deco if Timer.group_enable.get(group, True) else raw

    return apply


def timer_dynamic(group: str, **timer_kwargs) -> Callable:
    """Adds a timer with minimal overhead on nested functions.

    This should only be used for functions defined in local scopes (e.g., inside tests).

    Args:
        group: The group to time.
        timer_kwargs: The keyword arguments to pass to the timer.

    Returns:
        The wrapped function.
    """

    def apply(fn):
        raw = fn
        deco = _wrap_with_timer(fn, group, timer_kwargs)  # same helper you already have
        target = [deco if Timer.group_enable.get(group, True) else raw]  # mutable cell

        @functools.wraps(fn)
        def proxy(*a, **k):
            return target[0](*a, **k)

        _dynamic_func_registry.setdefault(group, []).append({"target": target, "raw": raw, "deco": deco})
        return proxy

    return apply


# --- togglers --------------------------------------------------------------------------------------------------------
def _set_timer_group_enabled_methods(cls: type, group: str, enabled: bool) -> None:
    """Swap all registered CLASS METHODS in 'group' on the given class."""
    try:
        group_map = cls._timer_toggle_registry[group]
    except (AttributeError, KeyError):
        return
    for name, (raw, deco) in group_map.items():
        setattr(cls, name, deco if enabled else raw)


def _set_timer_group_enabled_functions(group: str, enabled: bool) -> None:
    """Rebind all registered free functions in 'group' in their modules.

    Args:
        group: The group to toggle.
        enabled: Whether to enable the group.
    """
    mapping = _func_toggle_registry.get(group, {})
    for (mod_name, func_name), (raw, deco) in mapping.items():
        mod = sys.modules.get(mod_name)
        if mod:
            setattr(mod, func_name, deco if enabled else raw)


def _toggle_dynamic_functions(group: str, enabled: bool):
    """Toggle the dynamic functions for a group.

    Args:
        group: The group to toggle.
        enabled: Whether to enable the group.
    """
    entries = _dynamic_func_registry.get(group, [])
    for entry in entries:
        entry["target"][0] = entry["deco"] if enabled else entry["raw"]


def toggle_timer_group(group: str, enabled: bool) -> None:
    """Toggle a group across:
     - All classes that have timer methods (via metaclass/descriptor registry)
     - All registered free functions

    Args:
        group: The group to toggle.
        enabled: Whether to enable the group.
    """
    Timer.set_group(group, enabled)

    # classes
    for cls in list(_class_group_registry.get(group, ())):  # WeakSet -> snapshot
        _set_timer_group_enabled_methods(cls, group, enabled)

    # free functions
    _set_timer_group_enabled_functions(group, enabled)

    # dynamic functions
    _toggle_dynamic_functions(group, enabled)


def toggle_timer_group_display_output(group: str, enabled: bool) -> None:
    """Toggle the display output for a group.

    Args:
        group: The group to toggle.
        enabled: Whether to enable the display output.
    """
    Timer.set_group_display_output(group, enabled)
