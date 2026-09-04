# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-construction clone-plan dispatch and :class:`ReplicateSession` sugar."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from isaaclab.sim import SimulationContext

from .clone_plan import make_clone_plan
from .cloner_cfg import DEFAULT_ENV_TEMPLATE
from .cloner_strategies import sequential
from .usd import UsdReplicateContext

if TYPE_CHECKING:
    from .clone_plan import ClonePlan


REPLICATION_QUEUE: list[Any] = []
"""Constructed cfgs consumed by post-construction :func:`clone_plan_from_env_0` workflows.

Cfg-first :class:`ReplicateSession` planning does not read the queue. Dispatch clears it
without deriving any backend mapping from it.
"""


def queue_replication(cfg: Any) -> None:
    """Register a constructed cfg for post-construction clone planning.

    Args:
        cfg: Asset cfg with resolved ``prim_path``.
    """
    REPLICATION_QUEUE.append(cfg)


def replicate(plan: ClonePlan, *, replicate_physics: bool = True) -> None:
    """Dispatch a fully routed clone plan and publish it after replication.

    Planning derives routing from the input cfgs; dispatch does not rediscover or reshape that mapping.
    Every context is owned by the active :class:`~isaaclab.sim.SimulationContext` and receives
    only ``plan``. The queue is cleared up front, so a backend failure cannot leak stale entries
    into the next lifecycle.

    Args:
        plan: Replication layout to dispatch.
        replicate_physics: Whether physics replication clones each environment. If False,
            cloning is USD-only; an asset whose contexts are all physics-based is not cloned.
    """
    REPLICATION_QUEUE.clear()
    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("Clone-plan replication requires an active SimulationContext.")
    context_types = tuple(
        context_type for context_type in plan.context_rows if replicate_physics or context_type is UsdReplicateContext
    )
    missing = [context_type for context_type in context_types if context_type not in sim._backend_registry]
    if missing:
        names = ", ".join(f"{context_type.__module__}.{context_type.__qualname__}" for context_type in missing)
        raise RuntimeError(f"Clone contexts must be registered before plan dispatch: {names}.")

    contexts = [sim._backend_registry[context_type] for context_type in context_types]
    for context in sorted(contexts, key=lambda item: item.replicate_priority):
        context.replicate(plan)
    sim.set_clone_plan(plan)


class ReplicateSession:
    """Folds :func:`make_clone_plan` and :func:`replicate` into a ``with`` block.

    ``__enter__`` builds the complete plan and mutates each cfg's ``spawn_path``;
    ``__exit__`` clears constructor registrations and dispatches that same plan.

    Example:

        .. code-block:: python

            with cloner.ReplicateSession(cfgs, num_clones=128, env_spacing=2.0):
                for cfg in cfgs:
                    cfg.class_type(cfg)
    """

    def __init__(
        self,
        cfgs: Iterable[Any],
        num_clones: int,
        env_spacing: float,
        *,
        global_paths: tuple[str, ...] = (),
        clone_strategy: Callable[[np.ndarray, int], np.ndarray] = sequential,
        valid_set: np.ndarray | None = None,
        replicate_physics: bool = True,
        env_template: str = DEFAULT_ENV_TEMPLATE,
    ):
        """Capture arguments for :func:`make_clone_plan` and :func:`replicate`.

        Args:
            cfgs: Asset cfgs with resolved ``prim_path``.
            num_clones: Number of target envs.
            env_spacing: Grid spacing between env origins [m].
            global_paths: Complete shared-asset roots declared by the composition root. Defaults to none.
            clone_strategy: Prototype-to-env assignment function.
            valid_set: Optional ``[num_combos, num_groups]`` integer array of valid
                prototype combinations; ``None`` uses the full cartesian product.
            replicate_physics: Whether physics replication clones each environment;
                forwarded to :func:`replicate`.
            env_template: Path template for a replicated env prim, ``{}`` marking the env index.
        """
        self._cfgs = cfgs
        self._replicate_physics = replicate_physics
        self._kwargs = dict(
            num_clones=num_clones,
            env_spacing=env_spacing,
            global_paths=global_paths,
            clone_strategy=clone_strategy,
            valid_set=valid_set,
            env_template=env_template,
        )
        self._plan: ClonePlan | None = None

    def __enter__(self) -> ReplicateSession:
        self._plan = make_clone_plan(self._cfgs, **self._kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            assert self._plan is not None
            replicate(self._plan, replicate_physics=self._replicate_physics)
        else:
            # Drop cfgs registered before the failure so the next session is clean.
            REPLICATION_QUEUE.clear()

    @property
    def plan(self) -> ClonePlan:
        """The :class:`~isaaclab.cloner.ClonePlan` produced in :meth:`__enter__`."""
        if self._plan is None:
            raise RuntimeError("ReplicateSession.plan is only available inside the with block.")
        return self._plan
