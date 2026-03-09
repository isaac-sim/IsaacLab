# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab_experimental.managers.scene_entity_cfg import SceneEntityCfg

import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@wp.kernel
def _set_mask_from_ids(
    mask: wp.array(dtype=wp.bool),
    ids: wp.array(dtype=wp.int32),
):
    """Set ``mask[ids[i]] = True`` for each thread *i*."""
    i = wp.tid()
    mask[ids[i]] = True


def resolve_1d_mask(
    *,
    ids: Sequence[int] | slice | wp.array | torch.Tensor | None,
    mask: wp.array | torch.Tensor | None,
    all_mask: wp.array,
    scratch_mask: wp.array,
    device: str,
) -> wp.array:
    """Resolve ids/mask into a warp boolean mask.

    Matches the contract of ``ArticulationData._resolve_1d_mask`` on dev/newton.
    Callers must provide pre-allocated ``all_mask`` (all-True) and ``scratch_mask``
    (reusable working buffer). No allocations happen inside this function.

    Args:
        ids: Indices to set to ``True``. ``None`` or ``slice(None)`` means all.
        mask: Explicit boolean mask. If provided, returned directly (after
            torch→warp normalization if needed). Takes precedence over *ids*.
        all_mask: Pre-allocated all-True mask of shape ``(size,)``, returned
            when both *ids* and *mask* are ``None``.
        scratch_mask: Pre-allocated scratch mask of shape ``(size,)``, filled
            in-place when *ids* are provided.
        device: Warp device string.

    Returns:
        A ``wp.array(dtype=wp.bool)`` — ``mask``, ``all_mask``, or ``scratch_mask``.
    """
    # Fast path: explicit mask provided.
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
            if str(mask.device) != device:
                mask = mask.to(device)
            return wp.from_torch(mask, dtype=wp.bool)
        return mask

    # Fast path: all ids.
    if ids is None or (isinstance(ids, slice) and ids == slice(None)):
        return all_mask

    # Normalize slice into explicit indices.
    if isinstance(ids, slice):
        start, stop, step = ids.indices(scratch_mask.shape[0])
        ids = list(range(start, stop, step))
    elif not isinstance(ids, (torch.Tensor, wp.array)):
        ids = list(ids)

    # Prepare output mask.
    scratch_mask.fill_(False)

    # Normalize ids to wp.int32 array and launch kernel.
    if isinstance(ids, torch.Tensor):
        if ids.numel() == 0:
            return scratch_mask
        if str(ids.device) != device:
            ids = ids.to(device)
        if ids.dtype != torch.int32:
            ids = ids.to(dtype=torch.int32)
        if not ids.is_contiguous():
            ids = ids.contiguous()
        ids_wp = wp.from_torch(ids, dtype=wp.int32)
    elif isinstance(ids, wp.array):
        if ids.shape[0] == 0:
            return scratch_mask
        ids_wp = ids
    else:
        if len(ids) == 0:
            return scratch_mask
        ids_wp = wp.array(ids, dtype=wp.int32, device=device)

    wp.launch(kernel=_set_mask_from_ids, dim=ids_wp.shape[0], inputs=[scratch_mask, ids_wp], device=device)
    return scratch_mask


@wp.func
def wrap_to_pi(angle: float) -> float:
    """Wrap input angle (in radians) to the range [-pi, pi)."""
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


def resolve_asset_cfg(cfg: dict, env: ManagerBasedEnv) -> SceneEntityCfg:
    asset_cfg = None

    for value in cfg.values():
        # If it exists, the SceneEntityCfg should have been resolved by the base manager.
        if isinstance(value, SceneEntityCfg):
            asset_cfg = value
            # Check if the joint ids are not set, and if so, resolve them.
            if asset_cfg.joint_ids is None or asset_cfg.joint_ids == slice(None):
                asset_cfg.resolve_for_warp(env.scene)
            if asset_cfg.body_ids is None or asset_cfg.body_ids == slice(None):
                asset_cfg.resolve_for_warp(env.scene)
            break

    # If it doesn't exist, use the default robot entity.
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
        asset_cfg.resolve_for_warp(env.scene)

    return asset_cfg
