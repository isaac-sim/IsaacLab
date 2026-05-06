# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp frontend.

Routes a stable manager-based task cfg onto :class:`ManagerBasedRLEnvWarp`
by running a pluggable :class:`CompatRule` pipeline. Direct envs encode
their runtime in the env class itself (each warp direct env is its own
class), so for direct cfgs the frontend just verifies the registered
entry-point lives under :data:`WARP_ROOT_PREFIXES` and forwards to
:func:`gym.make`.

Rule pipeline (manager-based path):

* :class:`CheckPhysicsIsNewton` — blocking issue if ``cfg.sim.physics``
  is PhysX-flavoured, since the warp runtime needs Newton physics.
* :class:`ResolvePhysicsPreset` — collapses ``PresetCfg`` wrappers (no-op
  when Hydra already did so).
* :class:`DropUnsupportedSensors` — removes sensors warp can't run yet.
* :class:`PromoteSceneEntityCfg` — in-place class promotion to the warp
  variant so warp mdp kernels see ``joint_mask`` / ``joint_ids_wp``.
* :class:`SwapMdpFunctions` — name-based ``term.func`` replacement against
  the warp ``mdp`` modules.
* :class:`SwapActionClassType` — strict swap of action ``class_type`` to
  the warp twin.
* :class:`VerifyDirectIsWarp` — only fires for direct cfgs; blocks if
  the registered entry-point isn't a warp class.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable
from types import ModuleType
from typing import Any, ClassVar

from .base import (
    WARP_ROOT_PREFIXES,
    Change,
    CompatRule,
    Frontend,
    FrontendIncompatibleError,
    Issue,
    ResolveContext,
    Runtime,
    Severity,
    TaskMeta,
    Workflow,
    iter_term_attrs,
    resolve_warp_twin,
    walk_attrs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


class CheckPhysicsIsNewton(CompatRule):
    """Block if ``cfg.sim.physics`` isn't Newton.

    The warp runtime resolves asset / sensor ``class_type`` via the active
    physics backend's module tree (``isaaclab_newton.*`` for Newton,
    ``isaaclab_physx.*`` for PhysX). Loading a PhysX class under the warp
    runtime fails on ``omni.physics.tensors.api`` (a Kit module the warp
    runtime doesn't initialise), so a PhysX physics cfg is fundamentally
    incompatible with this frontend.

    Hydra's preset resolution + :meth:`WarpFrontend.preprocess_hydra_args`
    normally collapse ``PresetCfg`` wrappers to the ``newton`` field
    before this rule runs; we still check, so a programmatic caller or a
    misconfigured task fails fast and loudly here instead of crashing
    deep in scene init.
    """

    name = "check_physics_is_newton"

    NEWTON_MODULE_PREFIXES: ClassVar[tuple[str, ...]] = ("isaaclab_newton",)
    """Module prefixes that mark a physics cfg as Newton-flavoured."""

    PHYSX_MODULE_PREFIXES: ClassVar[tuple[str, ...]] = ("isaaclab_physx",)
    """Module prefixes that mark a physics cfg as PhysX-flavoured (definitely incompatible)."""

    def applies_to(self, ctx: ResolveContext) -> bool:
        return ctx.task.workflow == Workflow.MANAGER_BASED

    def run(self, cfg: Any, ctx: ResolveContext) -> Iterable[Issue | Change]:
        physics = getattr(getattr(cfg, "sim", None), "physics", None)
        if physics is None:
            return
        cls = type(physics)
        module = getattr(cls, "__module__", "") or ""
        # Skip PresetCfg wrappers — they'll be unwrapped by ResolvePhysicsPreset.
        if hasattr(physics, "newton") and not hasattr(physics, "class_type"):
            return
        if module.startswith(self.NEWTON_MODULE_PREFIXES):
            return  # OK
        severity = Severity.BLOCKING if module.startswith(self.PHYSX_MODULE_PREFIXES) else Severity.WARNING
        backend = "PhysX" if severity == Severity.BLOCKING else "unknown"
        yield Issue(
            rule=self.name,
            severity=severity,
            message=(
                f"sim.physics is {cls.__name__} ({module}); the warp runtime"
                f" needs a Newton physics cfg ({backend} detected). Pass"
                f" `presets=newton` or use a Newton physics cfg explicitly."
            ),
            location="cfg.sim.physics",
            detail={"physics_class": cls.__name__, "physics_module": module},
        )


class ResolvePhysicsPreset(CompatRule):
    """Collapse ``cfg.sim.physics`` from ``PresetCfg`` to its ``newton`` field.

    Hydra's preset resolution normally collapses the wrapper before the
    frontend runs. This rule covers the residual case of programmatic use
    of :meth:`Frontend.build` without Hydra, and custom physics cfgs that
    still expose a ``newton`` attribute.
    """

    name = "resolve_physics_preset"

    def applies_to(self, ctx: ResolveContext) -> bool:
        return ctx.task.workflow == Workflow.MANAGER_BASED

    def run(self, cfg: Any, ctx: ResolveContext) -> Iterable[Issue | Change]:
        physics = getattr(getattr(cfg, "sim", None), "physics", None)
        if physics is None or not hasattr(physics, "newton"):
            return
        cfg.sim.physics = physics.newton
        yield Change(
            rule=self.name,
            description=f"sim.physics → {type(physics.newton).__name__}",
        )


class DropUnsupportedSensors(CompatRule):
    """Drop scene sensors the warp runtime can't run yet.

    Default: ``("height_scanner",)``. Pass ``sensors=...`` at construction
    time to override.
    """

    name = "drop_unsupported_sensors"

    def __init__(self, sensors: Iterable[str] = ("height_scanner",)):
        self.sensors = tuple(sensors)

    def applies_to(self, ctx: ResolveContext) -> bool:
        return ctx.task.workflow == Workflow.MANAGER_BASED

    def run(self, cfg: Any, ctx: ResolveContext) -> Iterable[Issue | Change]:
        scene = getattr(cfg, "scene", None)
        if scene is None:
            return
        for sensor in self.sensors:
            if getattr(scene, sensor, None) is not None:
                setattr(scene, sensor, None)
                yield Change(rule=self.name, description=f"scene.{sensor} → None")


class PromoteSceneEntityCfg(CompatRule):
    """Promote :class:`SceneEntityCfg` instances under term params to the warp variant.

    The warp variant adds cached ``joint_mask`` / ``joint_ids_wp`` /
    ``body_ids_wp`` fields that the warp mdp kernels read after
    :meth:`resolve`. The class hierarchy is asserted at apply time — if
    the warp class no longer subclasses the stable class, the rule raises
    :class:`FrontendIncompatibleError` so the in-place ``__class__``
    promotion can never silently corrupt instances.
    """

    name = "promote_scene_entity_cfg"

    GROUPS: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("observations", "policy"),
        ("events",),
        ("rewards",),
        ("terminations",),
        ("commands",),
        ("curriculum",),
        ("actions",),
    )

    def applies_to(self, ctx: ResolveContext) -> bool:
        return ctx.task.workflow == Workflow.MANAGER_BASED

    def run(self, cfg: Any, ctx: ResolveContext) -> Iterable[Issue | Change]:
        from isaaclab.managers.scene_entity_cfg import SceneEntityCfg as Stable

        from isaaclab_experimental.managers.scene_entity_cfg import SceneEntityCfg as Warp

        if not issubclass(Warp, Stable):
            raise FrontendIncompatibleError(
                f"Warp SceneEntityCfg must subclass stable SceneEntityCfg; got mro {[c.__name__ for c in Warp.__mro__]}"
            )

        promoted = 0
        for path in self.GROUPS:
            group = walk_attrs(cfg, path)
            if group is None:
                continue
            for _name, term in iter_term_attrs(group):
                params = getattr(term, "params", None)
                if not isinstance(params, dict):
                    continue
                for value in params.values():
                    if isinstance(value, Warp) or not isinstance(value, Stable):
                        continue
                    # ``__class__ =`` is permitted only when the layouts
                    # match (e.g. neither side adds ``__slots__`` over the
                    # other). ``issubclass`` does *not* guarantee this. If
                    # the runtime refuses, surface a blocking issue rather
                    # than crash mid-pipeline; the cfg is still safe because
                    # we set warp-only fields *after* the assignment.
                    try:
                        value.__class__ = Warp
                    except TypeError as exc:
                        yield Issue(
                            rule=self.name,
                            severity=Severity.BLOCKING,
                            message=(
                                f"in-place __class__ promotion to {Warp.__name__} failed:"
                                f" {exc}. The warp variant likely diverged in slots/layout"
                                f" from the stable one; rebuild cfg explicitly with the warp"
                                f" SceneEntityCfg."
                            ),
                            location=".".join(path),
                            detail={"warp_class": Warp.__name__, "stable_class": Stable.__name__},
                        )
                        return
                    value.joint_mask = None
                    value.joint_ids_wp = None
                    value.body_ids_wp = None
                    promoted += 1
        if promoted:
            yield Change(
                rule=self.name,
                description=f"promoted {promoted} SceneEntityCfg instance(s) to warp variant",
            )


class SwapMdpFunctions(CompatRule):
    """Replace ``term.func`` with the same-named callable from warp ``mdp`` modules.

    Walks observation / event / reward / termination / command / curriculum
    groups. A re-export from stable code is rejected by checking
    ``__module__``. Missing twins are dropped (term set to ``None``) and
    surfaced as :attr:`Severity.WARNING`; in strict mode they're escalated
    to :attr:`Severity.BLOCKING`.
    """

    name = "swap_mdp_functions"

    GROUPS: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("observations", "policy"),
        ("events",),
        ("rewards",),
        ("terminations",),
        ("commands",),
        ("curriculum",),
    )

    def __init__(
        self,
        stable_root: str = "isaaclab_tasks",
        warp_root: str = "isaaclab_tasks_experimental",
        fallback_mdp: str = "isaaclab_experimental.envs.mdp",
    ):
        self.stable_root = stable_root
        self.warp_root = warp_root
        self.fallback_mdp = fallback_mdp

    def applies_to(self, ctx: ResolveContext) -> bool:
        return ctx.task.workflow == Workflow.MANAGER_BASED

    def run(self, cfg: Any, ctx: ResolveContext) -> Iterable[Issue | Change]:
        modules = self._mdp_modules(ctx.task)
        logger.info("WarpFrontend: warp mdp modules → %s", [m.__name__ for m in modules])
        searched = tuple(m.__name__ for m in modules)
        for path in self.GROUPS:
            group = walk_attrs(cfg, path)
            if group is None:
                continue
            location_prefix = ".".join(path)
            for name, term in iter_term_attrs(group):
                if not hasattr(term, "func"):
                    continue
                stable = term.func
                if not callable(stable):
                    continue
                # Idempotency: if the term already references a warp-native
                # callable (e.g. running the bridge against an already-warp
                # task), don't try to swap. Without this guard the rule would
                # ``resolve_warp_twin`` against modules that might not contain
                # the same symbol and silently drop the term.
                origin = getattr(stable, "__module__", "") or ""
                if origin.startswith(WARP_ROOT_PREFIXES):
                    continue
                twin = resolve_warp_twin(stable.__name__, modules)
                if twin is not None:
                    term.func = twin
                    continue
                # No twin found: drop the term and report.
                setattr(group, name, None)
                yield Issue(
                    rule=self.name,
                    severity=Severity.BLOCKING if ctx.strict else Severity.WARNING,
                    message=(f"no warp twin for stable func={stable.__name__!r}; term dropped."),
                    location=f"{location_prefix}.{name}",
                    detail={"expected": stable.__name__, "searched": list(searched)},
                )

    def _mdp_modules(self, task: TaskMeta) -> list[ModuleType]:
        """Locate warp mdp modules that mirror the task's stable mdp."""
        modules: list[ModuleType] = []
        entry = task.env_cfg_entry_point
        # Use a trailing dot when matching the stable root so a task registered
        # under ``isaaclab_tasks_experimental.*`` (the warp side) doesn't match
        # ``isaaclab_tasks`` and end up with a double-replaced
        # ``isaaclab_tasks_experimental_experimental.*`` import path.
        stable_prefix = self.stable_root + "."
        if isinstance(entry, str) and entry.startswith(stable_prefix):
            warp_pkg = entry.rsplit(".", 1)[0].replace(self.stable_root, self.warp_root, 1)
            parts = warp_pkg.split(".")
            for depth in range(len(parts), 0, -1):
                target = ".".join(parts[:depth] + ["mdp"])
                try:
                    modules.append(importlib.import_module(target))
                    break
                except ModuleNotFoundError as exc:
                    # Only swallow "this candidate module does not exist".
                    # An ImportError raised *inside* a module that does exist
                    # is a real bug we should surface, not paper over by
                    # falling back.
                    if exc.name == target:
                        continue
                    raise
        try:
            modules.append(importlib.import_module(self.fallback_mdp))
        except ModuleNotFoundError as exc:
            if exc.name != self.fallback_mdp:
                raise
            logger.warning("WarpFrontend: fallback mdp module %r not importable", self.fallback_mdp)
        return modules


class SwapActionClassType(CompatRule):
    """Swap ``cfg.actions.<term>.class_type`` with the same-named warp class.

    Always strict at the term level: an action with no warp twin can't run
    on the warp runtime, so a missing twin is :attr:`Severity.BLOCKING`
    regardless of ``ctx.strict``.
    """

    name = "swap_action_class_type"

    def __init__(self, warp_actions_module: str = "isaaclab_experimental.envs.mdp.actions.joint_actions"):
        self.warp_actions_module = warp_actions_module

    def applies_to(self, ctx: ResolveContext) -> bool:
        return ctx.task.workflow == Workflow.MANAGER_BASED

    def run(self, cfg: Any, ctx: ResolveContext) -> Iterable[Issue | Change]:
        actions = getattr(cfg, "actions", None)
        if actions is None:
            return
        try:
            module = importlib.import_module(self.warp_actions_module)
        except ImportError as exc:
            yield Issue(
                rule=self.name,
                severity=Severity.BLOCKING,
                message=(f"warp action module {self.warp_actions_module!r} not importable: {exc}"),
                location="actions",
            )
            return
        searched = (module.__name__,)
        for name, term in iter_term_attrs(actions):
            if not hasattr(term, "class_type"):
                continue
            stable = term.class_type
            if not callable(stable):
                continue
            twin = resolve_warp_twin(stable.__name__, [module])
            if twin is not None:
                term.class_type = twin
                continue
            yield Issue(
                rule=self.name,
                severity=Severity.BLOCKING,
                message=f"no warp twin for stable class_type={stable.__name__!r}",
                location=f"actions.{name}",
                detail={"expected": stable.__name__, "searched": list(searched)},
            )


class VerifyDirectIsWarp(CompatRule):
    """For direct envs, block if the registered entry-point isn't a warp class.

    Direct envs encode their runtime in the env class itself (each warp
    direct env is a separate class with its own kernels), so the warp
    frontend can't *adapt* a stable direct cfg — it can only refuse it
    with a clear message. A direct task whose entry-point already lives
    under :data:`WARP_ROOT_PREFIXES` is a pass-through.
    """

    name = "verify_direct_is_warp"

    def applies_to(self, ctx: ResolveContext) -> bool:
        return ctx.task.workflow == Workflow.DIRECT

    def run(self, cfg: Any, ctx: ResolveContext) -> Iterable[Issue | Change]:
        if ctx.task.runtime == Runtime.WARP:
            return
        ep = ctx.task.entry_point
        yield Issue(
            rule=self.name,
            severity=Severity.BLOCKING,
            message=(
                f"direct env entry_point {ep!r} is not a warp implementation."
                f" --frontend=warp on a direct task requires the registered class to live under"
                f" {list(WARP_ROOT_PREFIXES)} (e.g. *-Direct-Warp-v0)."
            ),
            location="task.entry_point",
            detail={"entry_point": ep},
        )


# ---------------------------------------------------------------------------
# WarpFrontend
# ---------------------------------------------------------------------------


class WarpFrontend(Frontend):
    """Run a stable env cfg on the warp runtime.

    Manager-based cfgs get the full rule pipeline and are constructed on
    :class:`ManagerBasedRLEnvWarp`. Direct cfgs run :class:`VerifyDirectIsWarp`
    only and dispatch through :func:`gym.make`.
    """

    name = "warp"
    rules = (
        CheckPhysicsIsNewton,
        ResolvePhysicsPreset,
        DropUnsupportedSensors,
        PromoteSceneEntityCfg,
        SwapMdpFunctions,
        SwapActionClassType,
        VerifyDirectIsWarp,
    )

    # -- construction ---------------------------------------------------------

    def construct(self, cfg: Any, meta: TaskMeta, **kwargs: Any) -> Any:
        if meta.workflow == Workflow.MANAGER_BASED:
            from isaaclab_experimental.envs import ManagerBasedRLEnvWarp

            return ManagerBasedRLEnvWarp(cfg=cfg, **kwargs)
        if meta.workflow == Workflow.DIRECT:
            import gymnasium as gym

            return gym.make(meta.task_id, cfg=cfg, **kwargs)
        raise FrontendIncompatibleError(
            f"WarpFrontend does not support workflow {meta.workflow.value!r} (task {meta.task_id!r})."
        )

    # -- CLI hook -------------------------------------------------------------

    def preprocess_hydra_args(self, task_id: str, args: list[str]) -> list[str]:
        """Inject ``presets=newton`` for stable manager-based tasks.

        Hydra resolves ``presets=...`` *before* the frontend runs, so the
        injection has to happen at the CLI layer. We only inject for tasks
        whose ``env_cfg_entry_point`` is under ``isaaclab_tasks.manager_based``;
        other tasks (direct, pre-warp ``*-Warp-v0``) don't carry a preset
        system and would error out.

        If the user already passed an explicit ``presets=`` we don't override,
        but we warn when their preset isn't ``newton``.
        """
        import gymnasium as gym

        try:
            spec = gym.spec(task_id)
        except gym.error.NameNotFound:
            return args
        cfg_ep = spec.kwargs.get("env_cfg_entry_point")
        if not isinstance(cfg_ep, str) or not cfg_ep.startswith("isaaclab_tasks.manager_based"):
            return args

        # Match both ``presets=foo`` and ``--presets=foo`` — Hydra accepts either.
        def _is_preset_arg(arg: str) -> bool:
            return arg.lstrip("-").startswith("presets=")

        explicit = next((a for a in args if _is_preset_arg(a)), None)
        if explicit is None:
            return [*args, "presets=newton"]
        if explicit.lstrip("-") != "presets=newton":
            logger.warning(
                "--frontend=warp on %r expects presets=newton; got %r — adapter may fail.",
                task_id,
                explicit,
            )
        return args
