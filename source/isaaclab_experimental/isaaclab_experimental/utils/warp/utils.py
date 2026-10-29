# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import warp as wp


def warp_capturable(capturable: bool):
    """Annotate an MDP term's CUDA-graph capturability.

    No-wrapper decorator: sets ``_warp_capturable`` directly on the function
    and returns it unchanged. Safe to stack with any other decorator in any order.

    By default all MDP terms are assumed capturable (True). Use
    ``@warp_capturable(False)`` on terms that call non-capturable external APIs.
    """

    def decorator(func):
        func._warp_capturable = capturable
        return func

    return decorator


def is_warp_capturable(func, params: dict[str, Any] | None = None) -> bool:
    """Check if a term is CUDA-graph-capturable.

    Checks ``_warp_capturable`` on the term function or class (the class of an instance) and on its
    ``__wrapped__`` target. The annotation is a bool, or a predicate that decides from the term's
    parameters (see :class:`WarpCapturable`). Returns True (capturable) if no annotation is found.

    Args:
        func: The term function, class, or class instance.
        params: The term's parameters, passed to a predicate annotation. Defaults to None (no parameters).
    """
    if not (isinstance(func, type) or inspect.isroutine(func)):
        func = type(func)
    for f in (func, getattr(func, "__wrapped__", None)):
        if f is not None:
            val = getattr(f, "_warp_capturable", None)
            if val is not None:
                return val if isinstance(val, bool) else bool(val(params or {}))
    return True


@wp.func
def wrap_to_pi(angle: float) -> float:
    """Wrap input angle (in radians) to the range [-pi, pi]."""
    two_pi = 2.0 * wp.pi
    wrapped_angle = angle + wp.pi
    # NOTE: Use floor-based remainder semantics to match torch's `%` for negative inputs.
    wrapped_angle = wrapped_angle - wp.floor(wrapped_angle / two_pi) * two_pi
    return wp.where((wrapped_angle == 0) and (angle > 0), wp.pi, wrapped_angle - wp.pi)


class WarpCapturable:
    """CUDA graph capture safety: decorator, annotation checker, and runtime guard.

    Decorator usage::

        @WarpCapturable(False)
        def reset_root_state_uniform(env, env_mask, ...):
            ...

        @WarpCapturable(False, reason="notifies the solver of a model change")
        class randomize_rigid_body_com(ManagerTermBase):
            ...

        @WarpCapturable(lambda params: params.get("sensor_cfg") is None, reason="refreshes the sensor on the host")
        def base_height_l2(env, out, target_height, asset_cfg=..., sensor_cfg=None):
            ...

    - ``@WarpCapturable(True)`` or no decorator: capturable, returned unwrapped.
    - ``@WarpCapturable(False)`` on a function: sets ``func._warp_capturable = False`` and wraps it with a
      runtime guard that raises if ``wp.get_device().is_capturing`` is ``True``.
    - ``@WarpCapturable(False)`` on a class term: annotates the class and guards its ``__call__``. The term's
      ``reset`` is still recorded; decorate the ``reset`` method itself to keep it out of the recording.
    - ``@WarpCapturable(predicate)``: the term is capturable for the parameters the predicate accepts. The
      predicate receives the term's parameters as a dict, and the guard raises only for rejected ones.
    """

    def __init__(self, capturable: bool | Callable[[dict[str, Any]], bool], *, reason: str | None = None):
        self._capturable = capturable
        self._reason = reason

    def __call__(self, target):
        """Decorate a term function or class with the capture safety annotation and runtime guard."""
        if isinstance(self._capturable, bool):
            target._warp_capturable = self._capturable
            if self._capturable:
                return target
        else:
            # a plain function stored on a class would bind to instances
            target._warp_capturable = staticmethod(self._capturable) if isinstance(target, type) else self._capturable
        if isinstance(target, type):
            target.__call__ = self._guarded(target.__call__)
            return target
        return self._guarded(target)

    def _guarded(self, func):
        """Wrap *func* so that calling it during CUDA graph capture with rejected parameters raises."""
        import functools

        capturable = self._capturable
        reason = self._reason

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if wp.get_device().is_capturing and (isinstance(capturable, bool) or not capturable(kwargs)):
                msg = (
                    f"'{func.__qualname__}' is not capturable with these parameters but was called during CUDA"
                    " graph capture."
                )
                if reason:
                    msg = f"{msg} {reason}"
                raise RuntimeError(msg)
            return func(*args, **kwargs)

        wrapper._warp_capturable = capturable
        return wrapper

    @staticmethod
    def is_capturable(func, params: dict[str, Any] | None = None) -> bool:
        """Check the capturability annotation. See :func:`is_warp_capturable`."""
        return is_warp_capturable(func, params)


@wp.kernel
def zero_masked_2d(mask: wp.array(dtype=wp.bool), values: wp.array(dtype=wp.float32, ndim=2)):
    """Zero out rows of a 2D float32 array where mask is True."""
    env_id, j = wp.tid()
    if mask[env_id]:
        values[env_id, j] = 0.0
