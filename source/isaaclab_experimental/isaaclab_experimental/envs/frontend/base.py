# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frontend framework — abstractions for runtime selection.

A *frontend* is a user-facing runtime selector (chosen via ``--frontend
{torch,warp}``). It takes a stable env cfg and a registered gym task id and
returns a runnable env, having first run a pluggable pipeline of
compatibility checks and transforms against the (cfg, task, frontend)
triple.

The framework is intentionally narrow:

* :class:`TaskResolver` is the single point that reads ``gym.spec`` and
  classifies a task into a :class:`TaskMeta` (workflow, registered
  runtime, cfg class). Rules and frontends consume :class:`TaskMeta`
  instead of poking gym directly, so the registration format can evolve
  in one place.
* :class:`CompatRule` is a single check or transform applied during
  resolution. Each rule yields :class:`Issue` records (incompatibilities
  the frontend can't paper over) and / or :class:`Change` records
  (transformations applied to the cfg). Subclasses implement
  :meth:`CompatRule.run` and may override :meth:`CompatRule.applies_to`
  to scope themselves by workflow / runtime.
* :class:`Frontend` is the dispatcher. Subclasses declare a name, a list
  of rule classes, and a :meth:`Frontend.construct` strategy that builds
  the env once rules pass.
* :func:`register_frontend` / :func:`get_frontend` form the registry CLI
  hooks read.

To add a new compatibility check, write a small :class:`CompatRule` and
list it in the relevant frontend's :attr:`Frontend.rules`. To add a new
runtime, subclass :class:`Frontend` and call :func:`register_frontend`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


WARP_ROOT_PREFIXES: tuple[str, ...] = ("isaaclab_experimental", "isaaclab_tasks_experimental")
"""Module prefixes that mark a class as a warp implementation.

The warp ``mdp`` packages do ``from isaaclab.envs.mdp import *``, so a
plain ``getattr`` lookup can return a stable symbol untouched. We accept
a candidate as a real warp twin only if its ``__module__`` lives under
one of these prefixes."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Workflow(str, Enum):
    """Task workflow type."""

    MANAGER_BASED = "manager_based"
    DIRECT = "direct"
    DIRECT_MARL = "direct_marl"


class Runtime(str, Enum):
    """Runtime backend a task is registered against.

    A task whose ``entry_point`` lives under :data:`WARP_ROOT_PREFIXES` is
    classified as ``WARP``. Tasks registered against ``isaaclab.envs.*``
    (the standard manager-based and direct env classes) are classified
    as ``TORCH``; those env classes use ``FactoryBase`` to dispatch on
    the active physics backend at construction time, so PhysX and Newton
    physics both flow through the torch runtime path.
    """

    TORCH = "torch"
    WARP = "warp"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Severity of a compatibility :class:`Issue`."""

    BLOCKING = "blocking"
    """Frontend cannot build the env. :meth:`Frontend.build` raises."""
    WARNING = "warning"
    """Surfaced to the user but not fatal (e.g. a term was dropped)."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class TaskMeta:
    """Static description of a registered gym task, produced by :class:`TaskResolver`."""

    task_id: str
    """Gym task id, e.g. ``Isaac-Cartpole-v0``."""
    spec: Any
    """The :class:`gymnasium.envs.registration.EnvSpec` returned by ``gym.spec``."""
    cfg_class: type
    """Concrete cfg type passed to the frontend."""
    workflow: Workflow
    """Manager-based / direct / direct-marl."""
    runtime: Runtime
    """Runtime the task is registered against (independent of the frontend asked for)."""

    @property
    def env_cfg_entry_point(self) -> str | None:
        """Module path of the cfg class as registered in ``spec.kwargs``."""
        return self.spec.kwargs.get("env_cfg_entry_point") if self.spec is not None else None

    @property
    def entry_point(self) -> Any:
        """Env class entry point as registered (string or class)."""
        return self.spec.entry_point if self.spec is not None else None


@dataclass
class Issue:
    """An incompatibility surfaced by a rule."""

    rule: str
    """Rule that produced this issue (matches :attr:`CompatRule.name`)."""
    severity: Severity
    message: str
    """One-line human-readable description."""
    location: str = ""
    """Optional cfg path the issue was found at (e.g. ``rewards.foo``)."""
    detail: dict[str, Any] = field(default_factory=dict)
    """Free-form rule-specific data (searched modules, expected names, ...)."""


@dataclass
class Change:
    """A transformation a rule applied to the cfg."""

    rule: str
    description: str


@dataclass
class Report:
    """Outcome of a frontend's rule pipeline."""

    frontend: str
    """Name of the frontend that produced this report."""
    task: TaskMeta
    rules_run: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)

    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.BLOCKING]

    def has_blocking(self) -> bool:
        return any(i.severity == Severity.BLOCKING for i in self.issues)

    def __bool__(self) -> bool:
        return bool(self.changes) or bool(self.issues)

    def format(self) -> str:
        lines = [
            f"Frontend {self.frontend!r} on task {self.task.task_id!r}"
            f" ({self.task.workflow.value} / registered runtime {self.task.runtime.value})"
        ]
        if self.changes:
            lines.append(f"  changes ({len(self.changes)}):")
            lines.extend(f"    [{c.rule}] {c.description}" for c in self.changes)
        if self.issues:
            lines.append(f"  issues ({len(self.issues)}):")
            for i in self.issues:
                head = f"    [{i.severity.value}/{i.rule}]"
                where = f" {i.location}" if i.location else ""
                lines.append(f"{head}{where} {i.message}")
        return "\n".join(lines)


@dataclass
class ResolveContext:
    """Per-call context handed to rules."""

    frontend: str
    """Name of the running frontend."""
    task: TaskMeta
    strict: bool = False
    """When True, rules may escalate warnings to blocking issues."""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FrontendIncompatibleError(RuntimeError):
    """Raised when a frontend cannot build the requested (task, cfg) pair."""

    def __init__(self, message: str, report: Report | None = None):
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# CompatRule
# ---------------------------------------------------------------------------


class CompatRule(ABC):
    """A single check / transform applied during frontend resolution.

    A rule's :meth:`run` method may yield :class:`Issue` records (the rule
    detected an incompatibility), :class:`Change` records (the rule mutated
    the cfg), or both. Pure-check and pure-transform rules are common; mixed
    rules are useful when a transformation has a partial-failure mode (e.g.
    a name swap that found 5 of 6 twins).

    Subclass and:

    * Set :attr:`name` to a short identifier used in the report.
    * Implement :meth:`run` to do the work.
    * Override :meth:`applies_to` to skip the rule when irrelevant.
    """

    name: ClassVar[str]

    def applies_to(self, ctx: ResolveContext) -> bool:
        """Return True if this rule should run on the given context.

        Default: always applies. Override to scope by ``ctx.task.workflow``,
        ``ctx.task.runtime``, ``ctx.frontend``, etc.
        """
        return True

    @abstractmethod
    def run(self, cfg: Any, ctx: ResolveContext) -> Iterable[Issue | Change]:
        """Inspect / mutate ``cfg`` and yield :class:`Issue` / :class:`Change` records."""


# ---------------------------------------------------------------------------
# TaskResolver
# ---------------------------------------------------------------------------


class TaskResolver:
    """Inspect a registered gym task and classify its workflow + runtime.

    All rules and frontends should call :meth:`resolve` rather than reading
    ``gym.spec`` directly. Centralising the read keeps registration-format
    knowledge in one place.
    """

    @classmethod
    def resolve(cls, task_id: str, cfg: Any) -> TaskMeta:
        """Return a :class:`TaskMeta` for ``(task_id, cfg)``."""
        import gymnasium as gym

        try:
            spec = gym.spec(task_id)
        except gym.error.NameNotFound:
            spec = None
        return TaskMeta(
            task_id=task_id,
            spec=spec,
            cfg_class=type(cfg),
            workflow=cls._classify_workflow(cfg),
            runtime=cls._classify_runtime(spec),
        )

    @staticmethod
    def _classify_workflow(cfg: Any) -> Workflow:
        from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

        if isinstance(cfg, ManagerBasedRLEnvCfg):
            return Workflow.MANAGER_BASED
        if isinstance(cfg, DirectMARLEnvCfg):
            return Workflow.DIRECT_MARL
        if isinstance(cfg, DirectRLEnvCfg):
            return Workflow.DIRECT
        raise FrontendIncompatibleError(
            f"Cfg type {type(cfg).__name__!r} is not a recognised env cfg"
            f" (expected ManagerBasedRLEnvCfg / DirectRLEnvCfg / DirectMARLEnvCfg subclass)."
        )

    @staticmethod
    def _classify_runtime(spec: Any) -> Runtime:
        if spec is None:
            return Runtime.UNKNOWN
        ep = spec.entry_point
        # ``entry_point`` may be a ``"module:Class"`` string (the common case)
        # or a class/callable object. Inspect ``__module__`` for the latter so
        # warp-registered tasks classified as such regardless of the
        # registration form.
        if isinstance(ep, str):
            module = ep
        else:
            module = getattr(ep, "__module__", "") or ""
        if module.startswith(WARP_ROOT_PREFIXES):
            return Runtime.WARP
        if module.startswith(("isaaclab.envs", "isaaclab_tasks")):
            return Runtime.TORCH
        return Runtime.UNKNOWN


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


class Frontend(ABC):
    """User-facing runtime selector.

    Subclasses declare:

    * :attr:`name`: CLI identifier used by ``--frontend`` and the registry.
    * :attr:`rules`: rule classes the pipeline instantiates by default.
    * :meth:`construct`: how to build the env once rules pass.

    Subclasses may also override:

    * :meth:`preprocess_hydra_args`: CLI-time Hydra arg munging (e.g. inject
      a preset selection so Hydra resolves ``PresetCfg`` to the right field
      before the cfg ever reaches us).
    * :meth:`preprocess_cfg`: cfg pre-processing run before rules.
    """

    name: ClassVar[str]
    rules: ClassVar[tuple[type[CompatRule], ...]] = ()

    def __init__(self, rules: Iterable[CompatRule] | None = None, strict: bool = False):
        self._rules: list[CompatRule] = [r() for r in type(self).rules] if rules is None else list(rules)
        self.strict = strict

    # -- public ---------------------------------------------------------------

    def build(self, cfg: Any, task_id: str, **construct_kwargs: Any) -> Any:
        """Resolve the task, run rules, and construct the env.

        Raises :class:`FrontendIncompatibleError` if any rule emits a
        :class:`Severity.BLOCKING` issue. The returned env carries the
        :class:`Report` on ``env.unwrapped.frontend_report`` for inspection.
        """
        report = self.resolve(cfg, task_id)
        self._log_report(report)
        if report.has_blocking():
            raise FrontendIncompatibleError(
                f"Frontend {self.name!r} cannot build {task_id!r}:\n{report.format()}",
                report=report,
            )
        env = self.construct(cfg, report.task, **construct_kwargs)
        env.unwrapped.frontend_report = report
        return env

    def resolve(self, cfg: Any, task_id: str) -> Report:
        """Run the rule pipeline; mutate ``cfg`` in place; return a :class:`Report`.

        Use this directly for dry-run validation (tests, CLI ``--check``).
        """
        meta = TaskResolver.resolve(task_id, cfg)
        ctx = ResolveContext(frontend=self.name, task=meta, strict=self.strict)
        report = Report(frontend=self.name, task=meta)
        # If gym couldn't resolve the spec, the frontend has nothing to dispatch
        # against. Block early with a clear message rather than letting
        # downstream construct() fail with a NameNotFound.
        if meta.spec is None:
            report.issues.append(
                Issue(
                    rule="task_resolver",
                    severity=Severity.BLOCKING,
                    message=(
                        f"task {task_id!r} is not registered with gymnasium."
                        " Make sure the task package is imported before the frontend runs."
                    ),
                    location="task.spec",
                )
            )
            return report
        self.preprocess_cfg(cfg, ctx)
        for rule in self._rules:
            if not rule.applies_to(ctx):
                continue
            report.rules_run.append(rule.name)
            for record in rule.run(cfg, ctx):
                if isinstance(record, Issue):
                    report.issues.append(record)
                elif isinstance(record, Change):
                    report.changes.append(record)
                else:
                    raise TypeError(f"Rule {rule.name!r} yielded {type(record).__name__}; expected Issue or Change.")
        return report

    # -- subclass hooks -------------------------------------------------------

    @abstractmethod
    def construct(self, cfg: Any, meta: TaskMeta, **kwargs: Any) -> Any:
        """Build the env. Called only when no blocking issues were raised."""

    def preprocess_cfg(self, cfg: Any, ctx: ResolveContext) -> None:
        """Subclass hook for cfg pre-processing run before rules. Default: no-op."""

    def preprocess_hydra_args(self, task_id: str, args: list[str]) -> list[str]:
        """Subclass hook for CLI-time Hydra arg munging. Default: no-op.

        Called from the train script so frontends can inject preset
        selections (or refuse incompatible ones) before Hydra builds the
        cfg. Returns the (possibly modified) args list.
        """
        return args

    # -- helpers --------------------------------------------------------------

    def _log_report(self, report: Report) -> None:
        if not report:
            return
        level = logging.ERROR if report.has_blocking() else logging.WARNING
        logger.log(level, report.format())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[Frontend]] = {}


def register_frontend(name: str, cls: type[Frontend]) -> None:
    """Register a frontend implementation under ``name``.

    The CLI ``--frontend`` flag and :func:`get_frontend` look up by this name.
    Re-registering the same class is idempotent; re-registering a different
    class raises :class:`ValueError`.
    """
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Frontend {name!r} is already registered to {existing.__name__};"
            f" refusing to re-register to {cls.__name__}."
        )
    _REGISTRY[name] = cls


def get_frontend(name: str, **kwargs: Any) -> Frontend:
    """Look up a frontend by name and return a fresh instance.

    Extra kwargs are forwarded to :meth:`Frontend.__init__` (e.g. ``strict=True``).
    """
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown frontend {name!r}; available: {available_frontends()}.") from exc
    return cls(**kwargs)


def available_frontends() -> list[str]:
    """Return registered frontend names, sorted."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Helpers shared by rules
# ---------------------------------------------------------------------------


def walk_attrs(root: Any, path: tuple[str, ...]) -> Any:
    """Walk ``root.<path[0]>.<path[1]>...`` returning ``None`` on any miss."""
    node = root
    for attr in path:
        node = getattr(node, attr, None)
        if node is None:
            return None
    return node


def resolve_warp_twin(name: str, modules: Iterable[Any]) -> Any | None:
    """Return ``name`` from ``modules`` if its ``__module__`` lives under a warp prefix."""
    for module in modules:
        candidate = getattr(module, name, None)
        if candidate is None:
            continue
        origin = getattr(candidate, "__module__", "") or ""
        if origin.startswith(WARP_ROOT_PREFIXES):
            return candidate
    return None


def iter_term_attrs(group: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, term)`` pairs from a manager-cfg group, skipping dunders."""
    if group is None:
        return
    for name in [n for n in dir(group) if not n.startswith("_")]:
        term = getattr(group, name, None)
        if term is None:
            continue
        yield name, term
