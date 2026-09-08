# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone-plan publication and dispatch."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from isaaclab.sim import SimulationContext

from .clone_plan import ClonePlan, make_clone_plan
from .cloner_cfg import DEFAULT_ENV_TEMPLATE
from .cloner_strategies import sequential
from .usd import UsdReplicateContext

REPLICATION_QUEUE: list[Any] = []
"""Constructed cfgs consumed by post-construction :func:`clone_plan_from_env_0` workflows.

Cfg-first :class:`ReplicateSession` planning does not read the queue. Dispatch clears it
without deriving any backend mapping from it.
"""


def queue_replication(cfg: Any) -> None:
    """Register a constructed cfg when no clone plan is active.

    Args:
        cfg: Asset cfg with resolved ``prim_path``.
    """
    if (sim := SimulationContext.instance()) is None or sim.get_clone_plan() is None:
        REPLICATION_QUEUE.append(cfg)


def replicate(plan: ClonePlan, *, replicate_physics: bool = True, isolate_environments: bool = True) -> None:
    """Publish and dispatch a fully routed clone plan.

    Planning derives routing from the input cfgs; dispatch does not rediscover or reshape that mapping.
    Every context is owned by the active :class:`~isaaclab.sim.SimulationContext` and receives
    only ``plan``. After argument validation, the queue is cleared before backend dispatch so a
    backend failure cannot leak stale entries into the next lifecycle.

    Args:
        plan: Replication layout to dispatch.
        replicate_physics: Whether physics replication clones each environment. If False,
            cloning is USD-only; an asset whose contexts are all physics-based is not cloned.
        isolate_environments: Whether collision participants in different environments are isolated.
            Shared assets declared by :attr:`~isaaclab.cloner.ClonePlan.global_paths` remain eligible
            to collide with every environment. Defaults to True.
    """
    if not isinstance(plan, ClonePlan):
        raise TypeError(f"plan must be a ClonePlan, got {type(plan).__name__}.")
    if not isinstance(replicate_physics, bool):
        raise TypeError("replicate_physics must be a bool.")
    if not isinstance(isolate_environments, bool):
        raise TypeError("isolate_environments must be a bool.")
    if plan.env_ids is None:
        raise ValueError("ClonePlan.env_ids is required for replication.")
    if (
        not isinstance(plan.env_ids, np.ndarray)
        or plan.env_ids.ndim != 1
        or not np.issubdtype(plan.env_ids.dtype, np.integer)
    ):
        raise TypeError("ClonePlan.env_ids must be a one-dimensional NumPy integer array.")
    if len(plan.sources) != len(plan.destinations):
        raise ValueError("ClonePlan.sources and ClonePlan.destinations must have equal length.")
    expected_shape = (len(plan.sources), len(plan.env_ids))
    if not isinstance(plan.clone_mask, np.ndarray) or plan.clone_mask.dtype != np.bool_:
        raise TypeError("ClonePlan.clone_mask must be a NumPy boolean array.")
    if plan.clone_mask.shape != expected_shape:
        raise ValueError(f"ClonePlan.clone_mask must have shape {expected_shape}, got {plan.clone_mask.shape}.")
    if plan.positions is not None:
        if not isinstance(plan.positions, np.ndarray):
            raise TypeError("ClonePlan.positions must be a NumPy array or None.")
        if plan.positions.shape != (len(plan.env_ids), 3):
            raise ValueError(
                f"ClonePlan.positions must have shape {(len(plan.env_ids), 3)}, got {plan.positions.shape}."
            )
    invalid_rows = sorted(
        {row for rows in plan.context_rows.values() for row in rows if row not in range(len(plan.sources))}
    )
    if invalid_rows:
        raise ValueError(f"ClonePlan.context_rows contains out-of-range rows: {invalid_rows}.")

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

    if (active_plan := sim.get_clone_plan()) is None:
        sim.set_clone_plan(plan)
    elif active_plan is not plan:
        raise ValueError("replicate() requires the active SimulationContext's ClonePlan.")

    contexts = sorted(
        (sim._backend_registry[context_type] for context_type in context_types),
        key=lambda item: item.replicate_priority,
    )
    stage_contexts = [context for context in contexts if context.replicate_priority < 0]
    physics_contexts = [context for context in contexts if context.replicate_priority >= 0]
    for context in stage_contexts:
        context.replicate(plan)
    sim.physics_manager.apply_collision_filter(
        plan,
        isolate_environments=isolate_environments,
        replicate_physics=replicate_physics,
    )
    for context in physics_contexts:
        context.replicate(plan)


class ReplicateSession:
    """Folds :func:`make_clone_plan` and :func:`replicate` into a ``with`` block.

    ``__enter__`` builds and publishes the complete plan while assigning each cfg's
    ``spawn_path``; ``__exit__`` dispatches that same plan.

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
        isolate_environments: bool = True,
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
            isolate_environments: Whether collision participants in different environments are isolated;
                forwarded to :func:`replicate`. Defaults to True.
            env_template: Path template for a replicated env prim, ``{}`` marking the env index.
        """
        self._cfgs = cfgs
        self._replicate_physics = replicate_physics
        self._isolate_environments = isolate_environments
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
        if (sim := SimulationContext.instance()) is None:
            raise RuntimeError("Clone planning requires an active SimulationContext.")
        if sim.get_clone_plan() is not None:
            raise RuntimeError("A SimulationContext owns exactly one clone lifecycle.")
        self._plan = make_clone_plan(self._cfgs, **self._kwargs)
        sim.set_clone_plan(self._plan)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            assert self._plan is not None
            replicate(
                self._plan,
                replicate_physics=self._replicate_physics,
                isolate_environments=self._isolate_environments,
            )
        else:
            # Drop cfgs registered before the failure so the next session is clean.
            REPLICATION_QUEUE.clear()
            if (sim := SimulationContext.instance()) is not None and sim.get_clone_plan() is self._plan:
                sim.set_clone_plan(None)

    @property
    def plan(self) -> ClonePlan:
        """The :class:`~isaaclab.cloner.ClonePlan` produced in :meth:`__enter__`."""
        if self._plan is None:
            raise RuntimeError("ReplicateSession.plan is only available inside the with block.")
        return self._plan
