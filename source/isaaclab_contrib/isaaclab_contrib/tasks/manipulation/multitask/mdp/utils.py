# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared config types and utilities for multitask MDP terms."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.managers import ManagerTermBaseCfg

ScatterResult = tuple[torch.Tensor | slice, torch.Tensor]
"""Return type for ``@scatterable`` functions: ``(env_ids, group_local_result)``."""


def _any_group_disabled(*args, **kwargs) -> bool:
    """Return True if any :class:`~isaaclab.managers.SceneEntityCfg` argument
    has an empty ``env_ids`` tensor, indicating its clone group is disabled.
    """
    for v in (*args, *kwargs.values()):
        if isinstance(v, SceneEntityCfg) and isinstance(v.env_ids, torch.Tensor) and v.env_ids.numel() == 0:
            return True
    return False


def scatterable(_func=None, /, *, output_dim: int | None = None, dtype: torch.dtype | None = None):
    """Decorator for group-aware MDP terms that produce partial-env results.

    The wrapped function returns ``(env_ids, result)`` where ``result``
    is group-local with shape ``(group_size, ...)``.  The decorator
    scatters into a full ``(num_envs, ...)`` buffer and returns a
    :class:`torch.Tensor`.

    Standalone calls reuse a persistent buffer (zero allocation after
    first call).  When called by :class:`scatter_term` with
    ``_out=buf``, the result is scattered into the provided buffer.

    Args:
        output_dim: Trailing output dimension for functions with a fixed output
            shape (e.g. ``output_dim=7`` for EE pose, ``output_dim=3`` for
            position error).  Use ``output_dim=0`` for scalar 1-D outputs
            (e.g. rewards or termination booleans).  Leave ``None`` for
            functions whose output size depends on asset configuration
            (e.g. ``joint_pos_rel`` where the count varies per robot).
        dtype: Element type of the persistent buffer (e.g. ``torch.bool`` for
            termination functions).  Inferred from the first real call result
            when ``None``.  Required when the group may be disabled before the
            first real call so the fast-path can allocate a correctly-typed
            zero buffer.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(env, *args, _out=None, **kwargs):
            # Fast-path: if any SceneEntityCfg arg has an empty env_ids tensor the group
            # is disabled — return the zero buffer without calling the inner function,
            # which would crash on a missing scene asset.
            if _any_group_disabled(*args, **kwargs):
                if _out is not None:
                    return _out
                if not hasattr(wrapper, "_buf"):
                    if wrapper.output_dim is not None:
                        # output_dim=0 means scalar 1-D output (e.g. terminations, rewards).
                        if wrapper.output_dim == 0:
                            wrapper._buf = torch.zeros(env.num_envs, dtype=wrapper.dtype, device=env.device)
                        else:
                            wrapper._buf = torch.zeros(
                                env.num_envs, wrapper.output_dim, dtype=wrapper.dtype, device=env.device
                            )
                    else:
                        # Variable-dim function with no output_dim and no prior call to
                        # infer shape.  Use MultiTaskObsTerm(dim=N) to give these a known
                        # fallback shape; _with_fallback_dim intercepts before this path.
                        return torch.zeros(env.num_envs, 1, device=env.device)
                _out = wrapper._buf
                _out.zero_()
                return _out

            env_ids, result = func(env, *args, **kwargs)
            if _out is None:
                if not hasattr(wrapper, "_buf"):
                    wrapper._buf = torch.zeros(env.num_envs, *result.shape[1:], dtype=result.dtype, device=env.device)
                _out = wrapper._buf
                _out.zero_()
            _out[env_ids] = result
            return _out

        wrapper.output_dim = output_dim
        wrapper.dtype = dtype
        return wrapper

    if _func is not None:
        # Called as bare @scatterable (no parentheses).
        return decorator(_func)
    return decorator


class scatter_term(ManagerTermBase):
    """Collects multiple ``@scatterable`` children into one output buffer.

    Pre-allocates a single ``(num_envs, D)`` buffer in ``__init__``.
    Each step: zeros the buffer, calls each child with ``_out=buf``
    so they scatter directly into it, returns the buffer.

    Children are :class:`ManagerTermBaseCfg` instances (just ``func`` +
    ``params``).  The outer term type (``ObsTerm``, ``RewTerm``, etc.)
    carries weights, noise, and other manager-specific fields.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._terms: list[ManagerTermBaseCfg] = cfg.params["terms"]
        output_dim = cfg.params.get("output_dim")
        if output_dim is None:
            output_dim = getattr(self._terms[0].func, "output_dim", None)
        if output_dim is None:
            raise ValueError(
                f"scatter_term could not determine output_dim. Either set "
                f"@scatterable(output_dim=D) on children or pass output_dim "
                f'in params: params={{"terms": [...], "output_dim": D}}. '
                f"Use output_dim=0 for scalar (1-D) outputs (e.g. rewards). "
                f"First child: {self._terms[0].func!r}"
            )
        if output_dim == 0:
            self._buf = torch.zeros(env.num_envs, device=env.device)
        else:
            self._buf = torch.zeros(env.num_envs, output_dim, device=env.device)

    def __call__(
        self, env: ManagerBasedRLEnv, terms: list | None = None, output_dim: int | None = None
    ) -> torch.Tensor:
        self._buf.zero_()
        for term_cfg in self._terms:
            term_cfg.func(env, **term_cfg.params, _out=self._buf)
        return self._buf


# ============================================================================
# Task filtering utilities
# ============================================================================


def _null_terms(cfg_block, predicate=None):
    """Set public attributes of *cfg_block* to ``None``.

    When *predicate* is given, only null where ``predicate(term)`` is True.
    """
    for attr in list(vars(cfg_block)):
        if attr.startswith("_"):
            continue
        term = getattr(cfg_block, attr)
        if term is not None and (predicate is None or predicate(term)):
            setattr(cfg_block, attr, None)


def apply_task_filter(env_cfg: ManagerBasedRLEnvCfg, disabled: set[str]) -> None:
    """Disable task selectors in *env_cfg* in-place.

    Sets the matching clone combination to ``weight=0``.  All downstream effects are now
    automatic — no further config patching is required:

    * **Assets** — :class:`~isaaclab.scene.InteractiveScene` does not spawn
      assets whose planned clone mask contains no environments.
    * **Commands** — :class:`~...mdp.commands.PoseCommand` detects the empty
      ``env_ids`` produced by :meth:`~isaaclab.managers.SceneEntityCfg.resolve`
      and enters a no-op mode without accessing the missing asset.
    * **Observations** — :class:`~...mdp.obs.MultiTaskObsTerm` wraps its
      ``func`` at construction time (via :meth:`__post_init__`) so that a
      correctly-shaped zero buffer is returned when the group is disabled.
    * **Rewards / terminations** — :func:`~...mdp.utils.scatterable` fast-paths
      on empty ``env_ids`` and returns zeros without touching missing assets.
    * **Events** — ``selector.filter_reset_ids`` returns empty indices for
      weight-zero selectors, causing event functions to return before any asset
      access.

    The only remaining manual step is clearing training-only MDP blocks
    (rewards, curriculum) that are meaningless during evaluation.

    Args:
        env_cfg: Top-level environment config to modify in place.
        disabled: Set of selector names to disable.
    """
    if not disabled:
        return

    # ── 1. Mark matching clone combinations as weight=0 ───────────────────
    for name in disabled:
        selector_term = getattr(env_cfg.scene.selector_cfg, name)
        selector_assets = set(selector_term.params["names"])
        for combination in env_cfg.scene.clone_cfg.clone_combinations:
            if set(combination.assets) == selector_assets:
                combination.weight = 0
                break

    # ── 2. Clear training-only MDP (rewards, curriculum) ──────────────────
    _null_terms(env_cfg.rewards)
    _null_terms(env_cfg.curriculum)
