# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone-plan publication and dispatch."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import numpy as np

from isaaclab.sim import SimulationContext

from .clone_plan import ClonePlan, make_clone_plan
from .cloner_cfg import CloneCfg
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


def replicate(plan: ClonePlan) -> None:
    """Publish and dispatch a fully routed clone plan.

    Planning derives routing from the input cfgs; dispatch does not rediscover or reshape that mapping.
    Every context is owned by the active :class:`~isaaclab.sim.SimulationContext` and receives
    only ``plan``. After argument validation, the queue is cleared before backend dispatch so a
    backend failure cannot leak stale entries into the next lifecycle.

    Args:
        plan: Replication layout to dispatch.
    """
    if not isinstance(plan, ClonePlan):
        raise TypeError(f"plan must be a ClonePlan, got {type(plan).__name__}.")
    if not isinstance(plan.isolate_environments, bool):
        raise TypeError("ClonePlan.isolate_environments must be a bool.")
    if not isinstance(plan.replicate_physics, bool):
        raise TypeError("ClonePlan.replicate_physics must be a bool.")
    REPLICATION_QUEUE.clear()
    sim = SimulationContext.instance()
    if sim is None:
        raise RuntimeError("Clone-plan replication requires an active SimulationContext.")
    context_types = tuple(
        context_type
        for context_type in plan.context_rows
        if plan.replicate_physics or context_type is UsdReplicateContext
    )
    missing = [context_type for context_type in context_types if context_type not in sim._backend_registry]
    if missing:
        names = ", ".join(f"{context_type.__module__}.{context_type.__qualname__}" for context_type in missing)
        raise RuntimeError(f"Clone contexts must be registered before plan dispatch: {names}.")

    active_plan = sim.get_clone_plan()
    if active_plan is None:
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
    sim.physics_manager.apply_collision_filter(plan)
    for context in physics_contexts:
        context.replicate(plan)


class ReplicateSession:
    """Folds :func:`make_clone_plan` and :func:`replicate` into a ``with`` block.

    ``__enter__`` builds and publishes the complete plan while assigning each cfg's
    ``spawn_path``; ``__exit__`` dispatches that same plan.

    Example:

        .. code-block:: python

            with cloner.ReplicateSession(cfgs, num_clones=128, env_spacing=2.0, clone_cfg=cloner.CloneCfg()):
                for cfg in cfgs:
                    cfg.class_type(cfg)
    """

    def __init__(
        self,
        cfgs: Iterable[Any],
        num_clones: int,
        env_spacing: float,
        clone_cfg: CloneCfg,
        *,
        global_paths: tuple[str, ...] = (),
        valid_set: np.ndarray | None = None,
    ):
        """Capture arguments for :func:`make_clone_plan` and :func:`replicate`.

        Args:
            cfgs: Asset cfgs with resolved ``prim_path``.
            num_clones: Number of target envs.
            env_spacing: Grid spacing between env origins [m].
            clone_cfg: Cloner configuration that owns planning and dispatch policy.
            global_paths: Complete shared-asset roots declared by the composition root. Defaults to none.
            valid_set: Optional ``[num_combos, num_groups]`` integer array of valid
                prototype combinations; ``None`` uses the full cartesian product.
        """
        if not isinstance(clone_cfg, CloneCfg):
            raise TypeError(f"clone_cfg must be a CloneCfg, got {type(clone_cfg).__name__}.")
        clone_cfg.validate_config()
        # Planning is deferred until __enter__, so the session owns a stable policy snapshot.
        clone_cfg = copy.deepcopy(clone_cfg)
        self._cfgs = cfgs
        self._kwargs: dict[str, Any] = dict(
            num_clones=num_clones,
            env_spacing=env_spacing,
            global_paths=global_paths,
            valid_set=valid_set,
            clone_cfg=clone_cfg,
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
            replicate(self._plan)
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
