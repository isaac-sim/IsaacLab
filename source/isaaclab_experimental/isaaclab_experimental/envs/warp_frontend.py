# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a stable env cfg on the warp runtime via a pluggable rule pipeline.

The frontend classifies the cfg as manager-based or direct, runs the
applicable :class:`CompatRule` instances against it, and constructs the
matching warp env class. New incompatibilities (e.g. a new sensor type warp
can't run, a new term-cfg field that needs upgrading) are added by writing a
small :class:`CompatRule` subclass and passing it through the constructor —
no surgery in the dispatcher.

Module-level imports are deliberately limited to lightweight stdlib modules.
Warp library imports fire only inside :meth:`WarpFrontend.build`, which must
be called *after* :class:`SimulationApp` is alive, otherwise warp's lib load
races with Kit's USD/pxr extension initialisation.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility-rule framework
# ---------------------------------------------------------------------------


class CfgKind(str, Enum):
    """Workflow type of an env cfg the frontend is asked to adapt."""

    MANAGER_BASED = "manager_based"
    DIRECT = "direct"


_BOTH = frozenset({CfgKind.MANAGER_BASED, CfgKind.DIRECT})
_MANAGER_ONLY = frozenset({CfgKind.MANAGER_BASED})

_WARP_MODULE_PREFIXES: tuple[str, ...] = ("isaaclab_experimental", "isaaclab_tasks_experimental")
"""Module-path prefixes that distinguish a real warp twin from a stable
re-export. The warp ``mdp`` packages do ``from isaaclab.envs.mdp import *``,
so a plain ``getattr`` lookup can return the stable symbol untouched."""

_TERM_GROUPS_FOR_PROMOTION: tuple[tuple[str, ...], ...] = (
    ("observations", "policy"),
    ("events",),
    ("rewards",),
    ("terminations",),
    ("commands",),
    ("curriculum",),
    ("actions",),
)
"""Cfg groups walked when promoting :class:`SceneEntityCfg` instances. Includes
``actions`` because action params often carry a :class:`SceneEntityCfg`."""


class IncompatibleEnvError(RuntimeError):
    """Raised when the frontend can't run on the given cfg / task pair."""

    def __init__(self, message: str, report: CompatReport | None = None):
        super().__init__(message)
        self.report = report


@dataclass
class CompatIssue:
    """One thing the adapter looked for but couldn't resolve to a warp twin."""

    rule: str
    """Rule that produced this issue (matches :attr:`CompatRule.name`)."""
    location: str
    """Cfg path the issue was found at, e.g. ``rewards`` / ``actions``."""
    item: str
    """Attribute name on the group, e.g. ``track_lin_vel_xy_exp``."""
    expected: str
    """The symbol the rule searched for (the stable ``__name__``)."""
    searched: tuple[str, ...]
    """Module dotted-paths inspected, in order."""
    action: str
    """Outcome: ``"dropped"``, ``"raised"``, or ``"left-stable"``."""


@dataclass
class CompatReport:
    """Outcome of running the rule pipeline against a cfg.

    ``changes`` is a human-readable list of mutations applied; ``issues``
    records every term the adapter couldn't resolve. A clean run has both
    empty.
    """

    rules_applied: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)
    issues: list[CompatIssue] = field(default_factory=list)

    def has_issues(self) -> bool:
        return bool(self.issues)

    def has_fatal(self) -> bool:
        return any(i.action == "raised" for i in self.issues)

    def __bool__(self) -> bool:
        return bool(self.issues) or bool(self.changes)

    def format(self) -> str:
        lines: list[str] = []
        if self.changes:
            lines.append(f"WarpFrontend applied {len(self.changes)} change(s):")
            lines.extend(f"  - {c}" for c in self.changes)
        if self.issues:
            lines.append(f"WarpFrontend has {len(self.issues)} unresolved item(s):")
            for i in self.issues:
                lines.append(
                    f"  - [{i.rule}] {i.location}.{i.item}: expected {i.expected!r}"
                    f" (searched {list(i.searched)}) → {i.action}"
                )
        return "\n".join(lines) if lines else "WarpFrontend: no changes."


@dataclass
class AdaptContext:
    """Per-call context handed to each rule."""

    task_id: str
    kind: CfgKind
    strict: bool


class CompatRule(ABC):
    """One step in the warp adaptation pipeline.

    A rule mutates ``cfg`` in place and appends to ``report``. To handle a
    new incompatibility, subclass and implement :meth:`apply`. Rules declare
    which workflow they apply to via :attr:`applies_to`; the pipeline skips
    any rule whose ``applies_to`` doesn't include the cfg kind.
    """

    name: ClassVar[str]
    applies_to: ClassVar[frozenset[CfgKind]]

    @abstractmethod
    def apply(self, cfg: Any, ctx: AdaptContext, report: CompatReport) -> None:
        """Mutate ``cfg`` in place and append issues / changes to ``report``."""


# ---------------------------------------------------------------------------
# Helpers shared by rules
# ---------------------------------------------------------------------------


def _walk(root: Any, path: tuple[str, ...]) -> Any:
    node = root
    for attr in path:
        node = getattr(node, attr, None)
        if node is None:
            return None
    return node


def _resolve_warp_twin(name: str, modules: Sequence[ModuleType]) -> Any | None:
    """Return ``name`` from ``modules`` if it lives under a warp prefix, else None."""
    for module in modules:
        candidate = getattr(module, name, None)
        if candidate is None:
            continue
        origin = getattr(candidate, "__module__", "") or ""
        if origin.startswith(_WARP_MODULE_PREFIXES):
            return candidate
    return None


def _swap_named_attr(
    group: Any,
    location: str,
    attr: str,
    modules: Sequence[ModuleType],
    *,
    rule: str,
    strict: bool,
    report: CompatReport,
) -> None:
    """Replace ``term.<attr>`` with the same-named callable from ``modules``.

    For each public attribute on ``group`` whose value has a callable ``attr``
    field, find a same-named symbol in ``modules`` whose ``__module__`` lives
    under :data:`_WARP_MODULE_PREFIXES` and assign it. If no twin is found,
    drop the term (set to ``None``) when ``strict`` is False, raise
    :class:`LookupError` when True. Either way append a :class:`CompatIssue`.
    """
    if group is None:
        return
    searched = tuple(m.__name__ for m in modules)
    # Snapshot the attribute names so mutating ``group`` mid-loop is safe.
    for name in [n for n in dir(group) if not n.startswith("_")]:
        term = getattr(group, name, None)
        if term is None or not hasattr(term, attr):
            continue
        stable = getattr(term, attr)
        if not callable(stable):
            continue
        twin = _resolve_warp_twin(stable.__name__, modules)
        if twin is not None:
            setattr(term, attr, twin)
            continue
        issue = CompatIssue(
            rule=rule,
            location=location,
            item=name,
            expected=stable.__name__,
            searched=searched,
            action="raised" if strict else "dropped",
        )
        report.issues.append(issue)
        if strict:
            raise LookupError(
                f"WarpFrontend: cannot adapt {location}.{name} — no warp twin for "
                f"stable {attr}={stable.__name__!r}. Searched {list(searched)}."
            )
        setattr(group, name, None)


# ---------------------------------------------------------------------------
# Concrete rules
# ---------------------------------------------------------------------------


class ResolvePhysicsPresetRule(CompatRule):
    """Collapse ``cfg.sim.physics`` from a :class:`PresetCfg` to its ``newton`` field.

    Hydra's preset resolution normally collapses the wrapper before the
    frontend runs. This rule covers two residual cases: programmatic use of
    :meth:`WarpFrontend.adapt` (no Hydra), and custom physics cfgs that still
    expose a ``newton`` attribute.
    """

    name = "resolve_physics_preset"
    applies_to = _BOTH

    def apply(self, cfg, ctx, report):
        physics = getattr(getattr(cfg, "sim", None), "physics", None)
        if physics is not None and hasattr(physics, "newton"):
            cfg.sim.physics = physics.newton
            report.changes.append(f"sim.physics → {type(physics.newton).__name__}")


class DropUnsupportedSensorsRule(CompatRule):
    """Drop scene sensors the warp runtime can't run yet."""

    name = "drop_unsupported_sensors"
    applies_to = _BOTH

    def __init__(self, sensors: Iterable[str] = ("height_scanner",)):
        self.sensors = tuple(sensors)

    def apply(self, cfg, ctx, report):
        scene = getattr(cfg, "scene", None)
        if scene is None:
            return
        for sensor in self.sensors:
            if getattr(scene, sensor, None) is not None:
                setattr(scene, sensor, None)
                report.changes.append(f"scene.{sensor} → None")


class PromoteSceneEntityCfgRule(CompatRule):
    """Promote :class:`SceneEntityCfg` instances under term params to the warp variant.

    The warp variant adds cached ``joint_mask`` / ``joint_ids_wp`` /
    ``body_ids_wp`` fields the warp mdp kernels read after :meth:`resolve`.
    The class hierarchy is asserted at apply time — if the warp class no
    longer subclasses the stable class (refactor or divergence), the rule
    raises :class:`TypeError` so the assumption can never silently corrupt
    instances.
    """

    name = "promote_scene_entity_cfg"
    applies_to = _MANAGER_ONLY

    def apply(self, cfg, ctx, report):
        from isaaclab.managers.scene_entity_cfg import SceneEntityCfg as _StableSE

        from isaaclab_experimental.managers.scene_entity_cfg import SceneEntityCfg as _WarpSE

        if not issubclass(_WarpSE, _StableSE):
            raise TypeError(
                "PromoteSceneEntityCfgRule requires the warp SceneEntityCfg to subclass the stable one;"
                f" got mro {[c.__name__ for c in _WarpSE.__mro__]}"
            )

        promoted = 0
        for path in _TERM_GROUPS_FOR_PROMOTION:
            group = _walk(cfg, path)
            if group is None:
                continue
            for name in (n for n in dir(group) if not n.startswith("_")):
                term = getattr(group, name, None)
                if term is None:
                    continue
                params = getattr(term, "params", None)
                if not isinstance(params, dict):
                    continue
                for value in params.values():
                    if isinstance(value, _WarpSE) or not isinstance(value, _StableSE):
                        continue
                    value.__class__ = _WarpSE
                    value.joint_mask = None
                    value.joint_ids_wp = None
                    value.body_ids_wp = None
                    promoted += 1
        if promoted:
            report.changes.append(f"promoted {promoted} SceneEntityCfg instance(s) to warp variant")


class SwapMdpFunctionsRule(CompatRule):
    """Replace ``term.func`` with the same-named callable from warp ``mdp`` modules.

    Walks observation / event / reward / termination / command / curriculum
    groups. A re-export from stable code is rejected by checking
    ``__module__``. Missing twins are dropped (term set to ``None``) unless
    the context is strict, in which case they raise.
    """

    name = "swap_mdp_functions"
    applies_to = _MANAGER_ONLY

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

    def apply(self, cfg, ctx, report):
        modules = self._mdp_modules_for(ctx.task_id)
        logger.info("WarpFrontend: warp mdp modules → %s", [m.__name__ for m in modules])
        for path in self.GROUPS:
            group = _walk(cfg, path)
            _swap_named_attr(
                group,
                location=".".join(path),
                attr="func",
                modules=modules,
                rule=self.name,
                strict=ctx.strict,
                report=report,
            )

    def _mdp_modules_for(self, task_id: str) -> list[ModuleType]:
        """Locate warp mdp modules that mirror the stable task's mdp.

        Convention: replace :attr:`stable_root` with :attr:`warp_root` in the
        stable cfg's module path, walk up looking for an ``mdp`` sub-module on
        the warp side, and always append :attr:`fallback_mdp` for cross-task
        base terms.
        """
        modules: list[ModuleType] = []
        try:
            import gymnasium as gym

            entry = gym.spec(task_id).kwargs.get("env_cfg_entry_point")
        except (KeyError, AttributeError, ModuleNotFoundError, gym.error.NameNotFound):
            # Spec is missing / kwargs missing the entry — fall back to the
            # package-wide warp mdp module only. Real ImportError from a
            # broken cfg module is *not* swallowed; it propagates.
            entry = None
        if isinstance(entry, str) and entry.startswith(self.stable_root):
            warp_pkg = entry.rsplit(".", 1)[0].replace(self.stable_root, self.warp_root, 1)
            parts = warp_pkg.split(".")
            for depth in range(len(parts), 0, -1):
                try:
                    modules.append(importlib.import_module(".".join(parts[:depth] + ["mdp"])))
                    break
                except ImportError:
                    continue
        try:
            modules.append(importlib.import_module(self.fallback_mdp))
        except ImportError:
            logger.warning("WarpFrontend: fallback mdp module %r not importable", self.fallback_mdp)
        return modules


class SwapActionClassTypeRule(CompatRule):
    """Swap ``cfg.actions.<term>.class_type`` with the same-named warp class.

    Always strict: an action with no warp class can't run on the warp
    runtime, so a missing twin raises :class:`LookupError`.
    """

    name = "swap_action_class_type"
    applies_to = _MANAGER_ONLY

    def __init__(self, warp_actions_module: str = "isaaclab_experimental.envs.mdp.actions.joint_actions"):
        self.warp_actions_module = warp_actions_module

    def apply(self, cfg, ctx, report):
        actions = getattr(cfg, "actions", None)
        if actions is None:
            return
        try:
            module = importlib.import_module(self.warp_actions_module)
        except ImportError as exc:
            raise IncompatibleEnvError(
                f"WarpFrontend: warp action module {self.warp_actions_module!r} not importable"
            ) from exc
        _swap_named_attr(
            actions,
            location="actions",
            attr="class_type",
            modules=[module],
            rule=self.name,
            strict=True,
            report=report,
        )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


class WarpFrontend:
    """Run a stable env cfg on the warp runtime via a pluggable rule pipeline.

    The frontend classifies the input cfg as manager-based or direct, runs
    the applicable :class:`CompatRule` instances, then constructs the
    matching warp env class.

    Direct envs encode their runtime in the env class itself (each warp
    direct env is a separate class with its own kernels), so the frontend
    doesn't transform direct cfgs — it only verifies the registered
    ``entry_point`` lives under :data:`_WARP_MODULE_PREFIXES` and dispatches
    via :func:`gym.make`. A stable direct cfg + ``--frontend=warp`` raises
    :class:`IncompatibleEnvError` with a clear message.

    Example::

        with launch_simulation(cfg, args_cli):
            env = WarpFrontend().build(cfg, task_id=args_cli.task)
            if env.unwrapped.warp_compat_report.has_issues():
                print(env.unwrapped.warp_compat_report.format())
    """

    DEFAULT_RULES: ClassVar[tuple[type[CompatRule], ...]] = (
        ResolvePhysicsPresetRule,
        DropUnsupportedSensorsRule,
        PromoteSceneEntityCfgRule,
        SwapMdpFunctionsRule,
        SwapActionClassTypeRule,
    )

    def __init__(self, rules: Sequence[CompatRule] | None = None, strict: bool = False):
        self.rules: list[CompatRule] = list(rules) if rules is not None else [r() for r in self.DEFAULT_RULES]
        self.strict = strict

    # -- public ---------------------------------------------------------------

    def adapt(self, cfg: Any, task_id: str) -> CompatReport:
        """Run the rule pipeline; mutate ``cfg`` in place; return a report.

        The report is logged at ``WARNING`` (or ``ERROR`` in strict mode when
        any issue is fatal). Missing twins raise :class:`LookupError` only
        when ``strict=True``.
        """
        kind = self._classify(cfg)
        ctx = AdaptContext(task_id=task_id, kind=kind, strict=self.strict)
        report = CompatReport()
        for rule in self.rules:
            if kind not in rule.applies_to:
                continue
            rule.apply(cfg, ctx, report)
            report.rules_applied.append(rule.name)
        if report:
            level = logging.ERROR if self.strict and report.has_fatal() else logging.WARNING
            logger.log(level, report.format())
        return report

    def build(self, cfg: Any, task_id: str, render_mode: str | None = None):
        """Adapt ``cfg`` and construct a warp env.

        Returns an env instance with ``warp_compat_report`` attached on the
        unwrapped env. ``render_mode`` is forwarded so ``--video`` keeps
        working when the frontend is selected at the train-script level.
        """
        # Lazy: this is the first warp-lib load. The caller must already be
        # inside the SimulationApp context (i.e. inside ``launch_simulation``).
        from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

        report = self.adapt(cfg, task_id)
        if isinstance(cfg, ManagerBasedRLEnvCfg):
            from isaaclab_experimental.envs import ManagerBasedRLEnvWarp

            env = ManagerBasedRLEnvWarp(cfg=cfg, render_mode=render_mode)
        elif isinstance(cfg, DirectRLEnvCfg):
            self._verify_direct_warp(task_id)
            import gymnasium as gym

            env = gym.make(task_id, cfg=cfg, render_mode=render_mode)
        else:
            raise IncompatibleEnvError(
                f"WarpFrontend supports ManagerBasedRLEnvCfg or DirectRLEnvCfg, got {type(cfg).__name__}",
                report=report,
            )
        env.unwrapped.warp_compat_report = report
        return env

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _classify(cfg: Any) -> CfgKind:
        from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

        if isinstance(cfg, ManagerBasedRLEnvCfg):
            return CfgKind.MANAGER_BASED
        if isinstance(cfg, DirectRLEnvCfg):
            return CfgKind.DIRECT
        raise IncompatibleEnvError(
            f"WarpFrontend supports ManagerBasedRLEnvCfg or DirectRLEnvCfg, got {type(cfg).__name__}"
        )

    @staticmethod
    def _verify_direct_warp(task_id: str) -> None:
        import gymnasium as gym

        try:
            spec = gym.spec(task_id)
        except gym.error.NameNotFound as exc:
            raise IncompatibleEnvError(f"Task {task_id!r} is not registered with gymnasium") from exc
        ep = spec.entry_point
        if isinstance(ep, str) and not ep.startswith(_WARP_MODULE_PREFIXES):
            raise IncompatibleEnvError(
                f"Direct env {task_id!r} entry_point {ep!r} is not a warp implementation."
                f" --frontend=warp on a direct task requires the registered class to live under"
                f" {list(_WARP_MODULE_PREFIXES)} (e.g. *-Direct-Warp-v0)."
            )
