# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replication queue, :func:`replicate` drain, and :class:`ReplicateSession` sugar."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from isaaclab.utils.backend_utils import FactoryBase
from isaaclab.utils.string import string_to_callable
from isaaclab.utils.version import has_kit

from . import path
from .clone_plan import make_clone_plan
from .cloner_cfg import DEFAULT_ENV_TEMPLATE
from .cloner_strategies import sequential
from .usd import UsdReplicateContext

if TYPE_CHECKING:
    import torch

    from pxr import Usd

    from .clone_plan import ClonePlan


REPLICATION_QUEUE: list[Any] = []
"""Asset cfgs registered by :func:`queue_replication` and drained by :func:`replicate`.

The queue only records *which* cfgs participate in cloning; how each cfg is cloned is
resolved at dispatch from :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` or the
active backend's default stack.
"""


def queue_replication(cfg: Any) -> None:
    """Register ``cfg`` for cloning when :func:`replicate` next runs.

    Args:
        cfg: Asset cfg with resolved ``prim_path``.
    """
    REPLICATION_QUEUE.append(cfg)


def replicate(plan: ClonePlan, *, stage: Usd.Stage, replicate_physics: bool = True) -> None:
    """Drain :data:`REPLICATION_QUEUE` against ``plan``, dispatch each backend, publish the plan.

    Physics contexts come from :attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` when
    set, otherwise from the backend's ``PHYSICS_CONTEXT`` class.
    :class:`~isaaclab.cloner.UsdReplicateContext` is added automatically when the cfg has a spawner
    and Kit is available. With physics replication enabled, this implicit context runs at
    :attr:`~isaaclab.physics.PhysicsEvent.MODEL_INIT`; explicit contexts remain synchronous.
    With ``replicate_physics=False`` physics contexts are dropped and USD replication remains
    synchronous, whether implicit or explicit.

    Cfgs absent from ``plan.cfg_rows`` are silently skipped. Synchronous backend contexts run
    ascending ``replicate_priority`` order. The queue is cleared up front, so a backend
    failure cannot leak stale entries into the next call.

    Args:
        plan: Replication layout to dispatch.
        stage: USD stage to author replicated prim specs into.
        replicate_physics: Whether physics replication clones each environment. If False,
            cloning is USD-only; an asset whose contexts are all physics-based is not cloned.
    """
    from isaaclab.physics import PhysicsEvent  # noqa: PLC0415
    from isaaclab.sim import SimulationContext  # noqa: PLC0415

    queued = REPLICATION_QUEUE.copy()
    REPLICATION_QUEUE.clear()

    backend_package = FactoryBase._get_package_name(FactoryBase._get_backend())
    backend_physics_ctx = getattr(importlib.import_module(f"{backend_package}.cloner"), "PHYSICS_CONTEXT", None)

    # Group queued cfgs by backend, taking the union of row indices each backend owns.
    # In the homogeneous plan every cfg maps to row 0, so multiple queue_replication
    # calls (e.g. one per body type in RigidObjectCollection) all contribute {0} and the set
    # union keeps it as a single row — no redundant copy specs are authored.
    kit_available = has_kit()
    backend_rows: dict[tuple[type, bool], set[int]] = {}
    for cfg in queued:
        rows = plan.cfg_rows.get(id(cfg))
        if rows is None:
            continue
        if cfg.cloning_contexts is None:
            contexts = [backend_physics_ctx] if backend_physics_ctx else []
        else:
            contexts = [string_to_callable(c) if isinstance(c, str) else c for c in cfg.cloning_contexts]
        if not replicate_physics:
            contexts = [c for c in contexts if c is UsdReplicateContext]
        ctx_set = dict.fromkeys(contexts)
        for BackendCtxCls in ctx_set:
            backend_rows.setdefault((BackendCtxCls, False), set()).update(rows)
        if getattr(cfg, "spawn", None) is not None and kit_available and UsdReplicateContext not in ctx_set:
            backend_rows.setdefault((UsdReplicateContext, replicate_physics), set()).update(rows)

    # One synchronous USD row already covers every cfg mapped to that row (notably the
    # homogeneous env-root row), so do not enqueue the same row again for MODEL_INIT.
    synchronous_usd_rows = backend_rows.get((UsdReplicateContext, False), set())
    deferred_usd_rows = backend_rows.get((UsdReplicateContext, True))
    if deferred_usd_rows is not None:
        deferred_usd_rows.difference_update(synchronous_usd_rows)
        if not deferred_usd_rows:
            del backend_rows[(UsdReplicateContext, True)]

    backend_ctxs: list[tuple[Any, bool]] = []
    for (BackendCtxCls, defer_to_model_init), row_set in backend_rows.items():
        ctx = BackendCtxCls(stage)
        backend_ctxs.append((ctx, defer_to_model_init))
        row_list = sorted(row_set)
        ctx.queue_mapping(
            [plan.sources[i] for i in row_list],
            [plan.destinations[i] for i in row_list],
            plan.env_ids,
            plan.clone_mask[row_list],
            positions=plan.positions,
        )

    sim = SimulationContext.instance()
    for ctx, defer_to_model_init in sorted(backend_ctxs, key=lambda item: getattr(item[0], "replicate_priority", 0)):
        if defer_to_model_init:
            sim.physics_manager.register_callback(ctx.replicate, PhysicsEvent.MODEL_INIT, order=2, wrap_weak_ref=False)
        else:
            ctx.replicate()

    sim.set_clone_plan(plan)


class ReplicateSession:
    """Folds :func:`make_clone_plan` and :func:`replicate` into a ``with`` block.

    ``__enter__`` builds the plan, positions its prototype environment roots, and mutates
    each cfg's ``spawn_path``. Asset constructors inside the block register their cfgs into
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
        replicate_physics: bool = True,
        env_template: str = DEFAULT_ENV_TEMPLATE,
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
            replicate_physics: Whether physics replication clones each environment;
                forwarded to :func:`replicate`.
            env_template: Path template for a replicated env prim, ``{}`` marking the env index.
        """
        self._cfgs = cfgs
        self._stage = stage
        self._replicate_physics = replicate_physics
        self._kwargs = dict(
            num_clones=num_clones,
            env_spacing=env_spacing,
            device=device,
            clone_strategy=clone_strategy,
            valid_set=valid_set,
            env_template=env_template,
        )
        self._plan: ClonePlan | None = None

    def __enter__(self) -> ReplicateSession:
        from pxr import Gf, UsdGeom  # noqa: PLC0415

        self._plan = make_clone_plan(self._cfgs, **self._kwargs)
        assert self._plan.env_ids is not None and self._plan.positions is not None
        assert self._plan.env_template is not None
        env_template = self._plan.env_template
        env_ids = self._plan.env_ids.cpu().tolist()
        positions = self._plan.positions.cpu().tolist()
        prototype_env_ids = {env_ids[0]}
        active_rows = self._plan.clone_mask.any(dim=1).cpu().tolist()
        for source, destination, is_active in zip(
            self._plan.sources, self._plan.destinations, active_rows, strict=True
        ):
            if not is_active:
                continue
            source_match = path.match(source, destination)
            assert source_match is not None and not source_match.suffix
            prototype_env_ids.add(int(source_match.instance))
        for env_id, position in zip(env_ids, positions, strict=True):
            if env_id in prototype_env_ids:
                root = UsdGeom.Xform.Define(self._stage, env_template.format(env_id))
                UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(*position))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            assert self._plan is not None
            replicate(self._plan, stage=self._stage, replicate_physics=self._replicate_physics)
        else:
            # Drop cfgs registered before the failure so the next session is clean.
            REPLICATION_QUEUE.clear()

    @property
    def plan(self) -> ClonePlan:
        """The :class:`~isaaclab.cloner.ClonePlan` produced in :meth:`__enter__`."""
        if self._plan is None:
            raise RuntimeError("ReplicateSession.plan is only available inside the with block.")
        return self._plan
