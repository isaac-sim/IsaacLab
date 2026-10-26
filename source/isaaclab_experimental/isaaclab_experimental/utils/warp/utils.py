# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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


def is_warp_capturable(func) -> bool:
    """Check if a term function is CUDA-graph-capturable.

    Checks ``_warp_capturable`` on the function and its ``__wrapped__`` target.
    Returns True (capturable) by default if no annotation is found.
    """
    for f in (func, getattr(func, "__wrapped__", None)):
        if f is not None:
            val = getattr(f, "_warp_capturable", None)
            if val is not None:
                return val
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

        @WarpCapturable(False, reason="calls write_root_pose_to_sim")
        def push_by_setting_velocity(env, env_mask, ...):
            ...

    - ``@WarpCapturable(True)`` or no decorator: capturable, returned unwrapped.
    - ``@WarpCapturable(False)``: sets ``func._warp_capturable = False``, wraps with
      runtime guard that raises if ``wp.get_device().is_capturing`` is ``True``.
    """

    def __init__(self, capturable: bool, *, reason: str | None = None):
        self._capturable = capturable
        self._reason = reason

    def __call__(self, func):
        """Decorate *func* with capture safety annotation and optional runtime guard."""
        import functools

        func._warp_capturable = self._capturable
        if self._capturable:
            return func

        reason = self._reason

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if wp.get_device().is_capturing:
                msg = f"'{func.__qualname__}' is marked @WarpCapturable(False) but called during CUDA graph capture."
                if reason:
                    msg = f"{msg} {reason}"
                raise RuntimeError(msg)
            return func(*args, **kwargs)

        wrapper._warp_capturable = False
        return wrapper

    @staticmethod
    def is_capturable(func) -> bool:
        """Check capturability annotation. Default: ``True``.

        Checks ``__wrapped__`` for decorated functions to handle stacked decorators.
        """
        for f in (func, getattr(func, "__wrapped__", None)):
            if f is not None:
                val = getattr(f, "_warp_capturable", None)
                if val is not None:
                    return val
        return True


@wp.kernel
def zero_masked_2d(mask: wp.array(dtype=wp.bool), values: wp.array(dtype=wp.float32, ndim=2)):
    """Zero out rows of a 2D float32 array where mask is True."""
    env_id, j = wp.tid()
    if mask[env_id]:
        values[env_id, j] = 0.0
