# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replication queue, :func:`replicate` drain, and :class:`ReplicateSession` sugar."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from .cloner_strategies import sequential

if TYPE_CHECKING:
    import torch

    from pxr import Usd

    from .clone_plan import ClonePlan


REPLICATION_QUEUE: list[tuple[Any, type]] = []
"""``(cfg, BackendCtxCls)`` pairs appended by ``queue_<backend>_replication`` and drained by :func:`replicate`."""


def replicate(plan: ClonePlan, *, stage: Usd.Stage) -> None:
    """Drain :data:`REPLICATION_QUEUE` against ``plan``, dispatch each backend, publish the plan.

    Cfgs absent from ``plan.cfg_rows`` are silently skipped. Backend contexts run in
    ascending ``replicate_priority`` order. The queue is cleared up front, so a backend
    failure cannot leak stale entries into the next call.
    """
    from isaaclab.sim import SimulationContext  # noqa: PLC0415

    queued = list(REPLICATION_QUEUE)
    REPLICATION_QUEUE.clear()

    # Group queued cfgs by backend, taking the union of row indices each backend owns.
    # In the homogeneous plan every cfg maps to row 0, so multiple queue_<backend>_replication
    # calls (e.g. one per body type in RigidObjectCollection) all contribute {0} and the set
    # union keeps it as a single row — no redundant copy specs are authored.
    backend_rows: dict[type, set[int]] = {}
    for cfg, BackendCtxCls in queued:
        rows = plan.cfg_rows.get(id(cfg))
        if rows is None:
            continue
        backend_rows.setdefault(BackendCtxCls, set()).update(rows)

    backend_ctxs: dict[type, Any] = {}
    for BackendCtxCls, row_set in backend_rows.items():
        ctx = BackendCtxCls(stage)
        backend_ctxs[BackendCtxCls] = ctx
        row_list = sorted(row_set)
        ctx.queue_mapping(
            [plan.sources[i] for i in row_list],
            [plan.destinations[i] for i in row_list],
            plan.env_ids,
            plan.clone_mask[row_list],
            positions=plan.positions,
        )

    for ctx in sorted(backend_ctxs.values(), key=lambda c: getattr(c, "replicate_priority", 0)):
        ctx.replicate()

    SimulationContext.instance().set_clone_plan(plan)


class ReplicateSession:
    """Folds :func:`make_clone_plan` and :func:`replicate` into a ``with`` block.

    ``__enter__`` builds the plan (and mutates each cfg's ``spawn_path``); asset
    constructors inside the block register backend replication into
    :data:`REPLICATION_QUEUE`; ``__exit__`` drains and dispatches.

    Example:

        .. code-block:: python

            with cloner.ReplicateSession(cfgs, num_clones=128, env_spacing=2.0, device="cuda:0", stage=sim.stage):
                for cfg in cfgs:
                    cfg.class_type(cfg)
    """

    def __init__(
        self,
        cfgs: Iterable[Any],
        num_clones: int,
        env_spacing: float,
        device: str,
        *,
        stage: Usd.Stage,
        clone_strategy: Callable = sequential,
        valid_set: torch.Tensor | None = None,
    ):
        """Capture arguments for :func:`make_clone_plan` and :func:`replicate`.

        Args:
            cfgs: Asset cfgs with resolved ``prim_path``.
            num_clones: Number of target envs.
            env_spacing: Grid spacing between env origins [m].
            device: Torch device for plan tensors.
            stage: USD stage to author replicated prim specs into.
            clone_strategy: Prototype-to-env assignment function.
            valid_set: Optional ``[num_combos, num_groups]`` long tensor of valid
                prototype combinations; ``None`` uses the full cartesian product.
        """
        self._cfgs = cfgs
        self._stage = stage
        self._kwargs = dict(
            num_clones=num_clones,
            env_spacing=env_spacing,
            device=device,
            clone_strategy=clone_strategy,
            valid_set=valid_set,
        )
        self._plan: ClonePlan | None = None

    def __enter__(self) -> ReplicateSession:
        from .cloner_utils import make_clone_plan  # noqa: PLC0415

        self._plan = make_clone_plan(self._cfgs, **self._kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            assert self._plan is not None
            replicate(self._plan, stage=self._stage)
        else:
            # Drop cfgs registered before the failure so the next session is clean.
            REPLICATION_QUEUE.clear()

    @property
    def plan(self) -> ClonePlan:
        """The :class:`~isaaclab.cloner.ClonePlan` produced in :meth:`__enter__`."""
        if self._plan is None:
            raise RuntimeError("ReplicateSession.plan is only available inside the with block.")
        return self._plan
