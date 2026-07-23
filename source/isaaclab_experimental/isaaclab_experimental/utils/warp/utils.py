# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import torch
import warp as wp

SYNC_DEBUG_ENV_VAR = "ISAACLAB_SYNC_DEBUG"
"""Set ``ISAACLAB_SYNC_DEBUG=1`` to trap hidden GPU->host syncs in warp stages (CI/debug)."""


def sync_debug_enabled() -> bool:
    """Whether the sync-debug trap is requested for this process."""
    return os.environ.get(SYNC_DEBUG_ENV_VAR, "0") == "1"


def any_env_set(mask: torch.Tensor) -> bool:
    """Host predicate: does the boolean mask select any environment?

    This is the single sanctioned per-step host sync of the mask-native reset
    pipeline (one predicate instead of dispatching an empty reset). It suspends
    the sync-debug trap around itself so the exemption is code, not annotation.
    """
    if mask.is_cuda:
        prev = torch.cuda.get_sync_debug_mode()
        if prev != 0:
            torch.cuda.set_sync_debug_mode(0)
            try:
                return bool(mask.any().item())
            finally:
                torch.cuda.set_sync_debug_mode(prev)
    return bool(mask.any().item())


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

    The single capturability annotation (see ``is_capturable``). Decorator usage::

        @WarpCapturable(False, reason="calls write_root_pose_to_sim")
        def push_by_setting_velocity(env, env_mask, ...):
            ...

        @WarpCapturable(False, reason="notifies the solver of model changes")
        class randomize_rigid_body_com(ManagerTermBase):
            ...

    - ``@WarpCapturable(True)`` or no decorator: capturable, returned unwrapped.
    - ``@WarpCapturable(False)`` on a function: sets ``_warp_capturable = False`` and
      wraps it with a runtime guard that raises if called during CUDA graph capture.
    - ``@WarpCapturable(False)`` on a class term: annotates the class (so manager
      registration sees it) and guards its ``__call__``.
    """

    def __init__(self, capturable: bool, *, reason: str | None = None):
        self._capturable = capturable
        self._reason = reason

    def __call__(self, target):
        """Decorate a term function or class with the annotation and optional guard."""
        target._warp_capturable = self._capturable
        if self._capturable:
            return target
        if isinstance(target, type):
            target.__call__ = self._guarded(target.__call__)
            return target
        return self._guarded(target)

    def _guarded(self, func):
        """Wrap *func* so calling it during CUDA graph capture raises."""
        import functools

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
    def is_capturable(term) -> bool:
        """Check the capturability annotation on a term function, class, or instance.

        Checks ``__wrapped__`` for decorated functions to handle stacked decorators.
        Unannotated terms default to capturable; hidden syncs in them are trapped by
        the ``ISAACLAB_SYNC_DEBUG`` stage sweep and by CUDA graph capture itself.
        """
        for f in (term, getattr(term, "__wrapped__", None)):
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
