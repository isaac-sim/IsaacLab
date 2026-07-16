# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime selector for IsaacLab tasks (``--frontend {torch,warp}``).

Two responsibilities only:

1. Decide which runtime constructs the env (torch via ``gym.make``, or warp via
   :class:`ManagerBasedRLEnvWarp`).
2. For the warp runtime, adapt a stable manager-based cfg in place so the
   warp managers can consume it (Newton physics + warp-twin term funcs and
   action classes + warp variant of :class:`SceneEntityCfg`).

The adaptation is a single function (:func:`_adapt_cfg_for_warp`); there is no
plugin framework. All warp MDP twins (functions *and* classes) must exist for
the chosen task — a missing twin is a hard failure, not a silent drop.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable, Iterator
from enum import StrEnum
from types import ModuleType
from typing import Any

import gymnasium as gym

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg as _StableSceneEntityCfg

logger = logging.getLogger(__name__)


__all__ = [
    "Frontend",
    "FrontendIncompatibleError",
    "Workflow",
    "adapt_cfg_for_warp",
    "build",
    "register_mdp_route",
]


# ---------------------------------------------------------------------------
# Enums + error
# ---------------------------------------------------------------------------


class Frontend(StrEnum):
    """Runtime selector exposed by ``--frontend``."""

    TORCH = "torch"
    WARP = "warp"


class Workflow(StrEnum):
    """Manager-based vs direct env workflow."""

    MANAGER_BASED = "manager_based"
    DIRECT = "direct"


class FrontendIncompatibleError(RuntimeError):
    """Raised when the chosen frontend can't run the requested task."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# Prefixes used to decide whether a registered task's entry-point lives under
# the warp packages. Used by the direct-workflow guard.
_WARP_ROOT_PREFIXES: tuple[str, ...] = ("isaaclab_experimental", "isaaclab_tasks_experimental")

# Stable package prefixes mapped to the warp MDP module that twins their terms.
# The stable task package is organized by public task domain while the
# experimental package separates manager-based workflows by family, so the
# module trees are not one-to-one. The registry is populated by the experimental
# task packages themselves via :func:`register_mdp_route` at import time, so
# adding a new task family never requires editing this module.
_WARP_MDP_MODULE_ROUTES: dict[str, str] = {}

# Root package that provides task-specific warp MDP twins. Imported (lazily, at
# swap time) so its task packages run their register_mdp_route() calls even when
# the caller only imported the stable task package.
_TWIN_PROVIDER_PACKAGE = "isaaclab_tasks_experimental"


def register_mdp_route(stable_package: str, warp_mdp_module: str) -> None:
    """Register the warp MDP module that twins a stable package's terms.

    Experimental task packages call this at import time to declare where the
    warp twins of a stable task's MDP terms live. :func:`_swap_mdp` consults
    the registry to resolve task-specific twins for ``--frontend=warp``. Twin
    lookup is by symbol name on ``warp_mdp_module`` itself, mirroring how a
    stable cfg consumes its ``mdp`` namespace — so the module should re-export
    everything the task needs (task-specific twins plus the generic forwards).

    Two kinds of keys are useful:

    * A *task package* (e.g. ``"isaaclab_tasks.core.reach"``) routes every cfg
      class defined under it. This also covers terms whose stable functions are
      defined in core/shared packages but overridden by task-specific twins.
    * An *MDP package* (e.g. ``"isaaclab_tasks.core.locomotion.humanoid.mdp"``)
      routes symbols *defined* under it, which serves other tasks that borrow
      those terms.

    Args:
        stable_package: Stable package prefix to route (longest match wins).
        warp_mdp_module: Warp MDP module providing the twins, e.g.
            ``"isaaclab_tasks_experimental.manager_based.classic.cartpole.mdp"``.

    Raises:
        ValueError: If ``stable_package`` is already routed to a different module.
    """
    existing = _WARP_MDP_MODULE_ROUTES.get(stable_package)
    if existing is not None and existing != warp_mdp_module:
        raise ValueError(
            f"MDP route conflict for {stable_package!r}: already routed to {existing!r}, got {warp_mdp_module!r}."
        )
    _WARP_MDP_MODULE_ROUTES[stable_package] = warp_mdp_module


# Top-level cfg groups whose managers run warp-first. Only terms under these are
# adapted (SceneEntityCfg promotion + MDP twin swap). The event manager is
# warp-first too — it invokes term funcs with a Warp env-mask, so a stable event
# func (which expects torch ``env_ids``) breaks at runtime; its funcs must be
# swapped to warp twins. The curriculum, recorder and command managers run on the
# stable (torch) implementation, so their terms are left untouched. A stable term
# left on a warp manager would break, so a missing twin in these groups is a hard
# error; a stable term on a stable manager is correct, so those groups are skipped.
_WARP_MANAGED_GROUPS: frozenset[str] = frozenset({"observations", "rewards", "terminations", "actions", "events"})


def build(
    frontend: Frontend | str,
    env_cfg: Any,
    task_id: str,
    **construct_kwargs: Any,
) -> gym.Env:
    """Construct the env on the selected runtime.

    Args:
        frontend: ``"torch"`` (default IsaacLab path) or ``"warp"`` (warp managers
            + :class:`ManagerBasedRLEnvWarp` for manager-based tasks).
        env_cfg: Stable env cfg. Mutated in place when ``frontend == "warp"``.
        task_id: Gym registration id, e.g. ``"Isaac-Cartpole-v0"``.
        **construct_kwargs: Forwarded to the env constructor (``render_mode``, …).

    Returns:
        The constructed :class:`gym.Env`.

    Raises:
        FrontendIncompatibleError: If the warp runtime can't run the task
            (wrong physics, missing MDP twins, direct task not registered
            as a warp env).
    """
    frontend = Frontend(frontend)
    if frontend is Frontend.TORCH:
        return gym.make(task_id, cfg=env_cfg, **construct_kwargs)

    workflow = _detect_workflow(env_cfg)
    if workflow is Workflow.DIRECT:
        # Direct workflows aren't adapted — they must already be registered
        # under the warp packages (e.g. ``Isaac-Cartpole-Direct-Warp``).
        _assert_direct_warp_registration(task_id)
        return gym.make(task_id, cfg=env_cfg, **construct_kwargs)

    # Imported lazily so that ``--frontend=torch`` callers don't pay the
    # ``isaaclab_experimental.envs`` import cost. Registered ``*-Warp-v0`` tasks
    # already provide Warp-native configs; the frontend path adapts stable configs
    # immediately before constructing the Warp environment.
    from isaaclab_experimental.envs import ManagerBasedRLEnvWarp

    adapt_cfg_for_warp(env_cfg)
    return ManagerBasedRLEnvWarp(cfg=env_cfg, **construct_kwargs)


# ---------------------------------------------------------------------------
# Cfg adaptation (warp only)
# ---------------------------------------------------------------------------


def adapt_cfg_for_warp(cfg: Any) -> None:
    """Mutate a stable manager-based ``cfg`` in place so warp managers can consume it.

    Called by :func:`build` for ``--frontend=warp`` on a stable manager-based
    task. Idempotent: re-running on an already-adapted cfg is a no-op (the steps
    below skip Warp-origin symbols and already-promoted entities).

    Three steps, each independently testable:

    1. :func:`_require_newton_physics` — hard check that ``cfg.sim.physics`` is
       :class:`~isaaclab_newton.physics.NewtonCfg`. The user is responsible for
       selecting the Newton variant of the task's :class:`PresetCfg` via
       ``presets=newton_mjwarp``; we don't auto-inject.
    2. :func:`_promote_scene_entity_cfgs` — replace stable
       :class:`~isaaclab.managers.SceneEntityCfg` instances under each term's
       ``params`` with the warp variant (which adds warp-cached ``joint_mask``,
       ``joint_ids_wp``, ``body_ids_wp`` fields).
    3. :func:`_swap_mdp` — for every MDP term found anywhere in the cfg tree
       (discovered by :func:`_walk_terms` via :class:`ManagerTermBaseCfg`
       subclassing, not by hard-coded attribute names), replace any stable
       ``func`` *or* ``class_type`` with its same-named warp twin. A missing
       twin raises :class:`FrontendIncompatibleError` — partial coverage is
       unsafe under the warp managers' kernel-only signature.
    """
    label = type(cfg).__name__
    _require_newton_physics(cfg, label)
    _promote_scene_entity_cfgs(cfg)
    _swap_mdp(cfg, label)


def _require_newton_physics(cfg: Any, label: str) -> None:
    """Block unless ``cfg.sim.physics`` is :class:`NewtonCfg`.

    The warp managers' assets read state through :class:`NewtonManager`;
    a :class:`PhysxCfg` (or unresolved :class:`PresetCfg`) is a hard
    incompatibility. The fix is to pass ``presets=newton_mjwarp`` on the CLI so
    Hydra resolves the task's :class:`PresetCfg` wrapper to the Newton field
    before construction.
    """
    from isaaclab_newton.physics import NewtonCfg

    physics = getattr(getattr(cfg, "sim", None), "physics", None)
    if isinstance(physics, NewtonCfg):
        return
    raise FrontendIncompatibleError(
        f"warp env {label!r}: expected cfg.sim.physics to be NewtonCfg,"
        f" got {type(physics).__name__!r}. Pass `presets=newton_mjwarp` on the CLI so"
        f" Hydra resolves the task's PresetCfg wrapper to the Newton variant."
    )


def _promote_scene_entity_cfgs(cfg: Any) -> None:
    """Replace stable :class:`SceneEntityCfg` instances with the warp variant.

    Iterates every term cfg in the tree (via :func:`_walk_terms`) and rebuilds
    any stable :class:`SceneEntityCfg` value under ``term.params`` through
    :meth:`isaaclab_experimental.managers.SceneEntityCfg.from_stable`. The
    warp variant subclasses the stable one, so type checks elsewhere stay
    valid; the new fields (``joint_mask`` / ``joint_ids_wp`` / ``body_ids_wp``)
    are filled at :meth:`resolve` time by the warp scene.
    """
    from isaaclab_experimental.managers.scene_entity_cfg import SceneEntityCfg as _WarpSceneEntityCfg

    promoted: list[str] = []
    for path, term in _walk_terms(cfg):
        if not path or path[0] not in _WARP_MANAGED_GROUPS:
            continue
        params = getattr(term, "params", None)
        if not isinstance(params, dict):
            continue
        for key, value in list(params.items()):
            if isinstance(value, _WarpSceneEntityCfg) or not isinstance(value, _StableSceneEntityCfg):
                continue
            params[key] = _WarpSceneEntityCfg.from_stable(value)
            promoted.append(f"{'.'.join(path)}.params[{key!r}] ({value.name!r})")
    if promoted:
        logger.info(
            "frontend.warp: promoted %d SceneEntityCfg instance(s) to warp variant:\n  %s",
            len(promoted),
            "\n  ".join(promoted),
        )


def _swap_mdp(cfg: Any, label: str) -> None:
    """Replace ``term.func`` and ``term.class_type`` with their warp twins.

    Iterates every term cfg in the tree (via :func:`_walk_terms`) and on each
    term swaps whichever of ``func`` / ``class_type`` is a stable-origin
    callable. Twin lookup is name-based, in the order given by
    :func:`_twin_modules`: the warp mirror of the cfg's own task MDP namespace
    first (task twins win, even for symbols defined in core packages), then the
    mirror of the symbol's defining package (covers terms borrowed from another
    task family, e.g. the Ant task reusing Humanoid MDP terms), then the shared
    :mod:`isaaclab_experimental.envs.mdp` fallback. Any missing twin raises
    :class:`FrontendIncompatibleError` listing every affected term — partial
    swaps would leave torch-style callables in the cfg and the warp managers
    would call them with the wrong signature.

    The warp-side declarations (``out_dim``, ``axes``, ``observation_type``)
    that the warp managers need at init are *not* supplied by this swap; they
    travel with the warp twin function itself via its own
    ``@generic_io_descriptor_warp(out_dim=…)`` decorator. This function only
    substitutes the callable; the manager reads the new func's annotations
    when it parses the term cfg.
    """
    _ensure_twin_providers_imported()

    cfg_route_modules = _cfg_route_modules(cfg)
    module_cache: dict[str, list[ModuleType]] = {}
    searched: set[str] = set()

    swapped = 0
    missing: list[tuple[str, str, str]] = []  # (location, attr, symbol)
    for path, term in _walk_terms(cfg):
        if not path or path[0] not in _WARP_MANAGED_GROUPS:
            continue
        location = ".".join(path)
        for attr in ("func", "class_type"):
            stable = getattr(term, attr, None)
            if stable is None or not _is_swap_candidate(stable):
                continue
            origin = getattr(stable, "__module__", "") or ""
            if origin not in module_cache:
                module_cache[origin] = _twin_modules(origin, cfg_route_modules)
                searched.update(m.__name__ for m in module_cache[origin])
            twin = _resolve_warp_twin(stable.__name__, module_cache[origin])
            if twin is None:
                missing.append((location, attr, stable.__name__))
                continue
            setattr(term, attr, twin)
            swapped += 1

    if missing:
        lines = "\n  ".join(f"{loc}.{attr}: no warp twin for {sym!r}" for loc, attr, sym in missing)
        raise FrontendIncompatibleError(
            f"warp env {label!r}: missing warp MDP twins (searched {sorted(searched)}):\n  {lines}"
        )

    logger.info("frontend.warp: swapped %d MDP symbol(s) to warp twins", swapped)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_workflow(cfg: Any) -> Workflow:
    """Classify the env cfg into manager-based or direct (used to pick build path).

    Note:
        The four env cfg roots (ManagerBasedEnvCfg, ManagerBasedRLEnvCfg,
        DirectRLEnvCfg, DirectMARLEnvCfg) do not share a common base class.
        When a new cfg root is added, extend the isinstance tuples below.
    """
    if isinstance(cfg, ManagerBasedRLEnvCfg):
        return Workflow.MANAGER_BASED
    if isinstance(cfg, (DirectRLEnvCfg, DirectMARLEnvCfg)):
        return Workflow.DIRECT
    raise FrontendIncompatibleError(
        f"Unrecognised env cfg type {type(cfg).__name__!r};"
        f" expected ManagerBasedRLEnvCfg / DirectRLEnvCfg / DirectMARLEnvCfg subclass."
    )


def _assert_direct_warp_registration(task_id: str) -> None:
    """For direct workflows, the task must be pre-registered under the warp packages."""
    try:
        spec = gym.spec(task_id)
    except gym.error.NameNotFound as exc:
        raise FrontendIncompatibleError(f"--frontend=warp: task {task_id!r} is not registered with gymnasium.") from exc
    ep = spec.entry_point
    module = ep if isinstance(ep, str) else (getattr(ep, "__module__", "") or "")
    if not module.startswith(_WARP_ROOT_PREFIXES):
        raise FrontendIncompatibleError(
            f"--frontend=warp on direct task {task_id!r}: entry_point {ep!r}"
            f" is not under {list(_WARP_ROOT_PREFIXES)}. Direct tasks must be"
            f" registered as a warp env class (e.g. *-Direct-Warp-v0)."
        )


def _match_route(module: str) -> str | None:
    """Return the routed warp MDP module for ``module``, longest prefix wins."""
    if not isinstance(module, str) or not module:
        return None
    best_key: str | None = None
    for stable_prefix in _WARP_MDP_MODULE_ROUTES:
        if module == stable_prefix or module.startswith(f"{stable_prefix}."):
            if best_key is None or len(stable_prefix) > len(best_key):
                best_key = stable_prefix
    return _WARP_MDP_MODULE_ROUTES[best_key] if best_key is not None else None


def _import_routed_module(target: str) -> ModuleType:
    """Import a registered route target; a broken registration is a hard error."""
    try:
        return importlib.import_module(target)
    except ImportError as exc:
        raise FrontendIncompatibleError(f"registered warp MDP route target {target!r} failed to import: {exc}") from exc


def _cfg_route_modules(cfg: Any) -> list[ModuleType]:
    """Warp MDP modules routed from the cfg's class hierarchy, subclass first.

    A stable cfg consumes terms through its task's ``mdp`` namespace, which
    re-exports symbols that may be *defined* in core or shared packages. The
    warp mirror of that namespace is therefore the primary place to resolve
    twins, and it is found by routing the modules of ``type(cfg).__mro__`` —
    the subclass module first (robot-specific cfgs), then base-cfg modules
    (task-family bases).
    """
    modules: list[ModuleType] = []
    for klass in type(cfg).__mro__:
        target = _match_route(getattr(klass, "__module__", "") or "")
        if target is None:
            continue
        module = _import_routed_module(target)
        if module not in modules:
            modules.append(module)
    return modules


def _twin_modules(symbol_module: str, cfg_route_modules: list[ModuleType]) -> list[ModuleType]:
    """Warp modules to consult for a stable symbol's twin, in preference order.

    1. The warp mirrors of the cfg's own task MDP namespace
       (:func:`_cfg_route_modules`) — task-specific twins win, including twins
       for symbols defined in core/shared packages.
    2. The warp mirror routed from the symbol's defining package — covers terms
       borrowed from another task family's MDP package.
    3. The shared :mod:`isaaclab_experimental.envs.mdp` fallback (where generic
       warp twins live).
    """
    modules = list(cfg_route_modules)
    target = _match_route(symbol_module)
    if target is not None:
        module = _import_routed_module(target)
        if module not in modules:
            modules.append(module)
    fallback = "isaaclab_experimental.envs.mdp"
    try:
        fallback_module = importlib.import_module(fallback)
    except ModuleNotFoundError as exc:
        if exc.name != fallback:
            raise
        logger.warning("frontend.warp: fallback mdp module %r not importable", fallback)
    else:
        if fallback_module not in modules:
            modules.append(fallback_module)
    return modules


def _ensure_twin_providers_imported() -> None:
    """Import the twin-provider package so its MDP routes are registered.

    Route registration happens as an import side effect of the experimental
    task packages. A caller running ``--frontend=warp`` on a stable task id
    has no other reason to import :mod:`isaaclab_tasks_experimental`, so the
    swap triggers it here. A missing provider package is not an error by
    itself — the swap then fails with the explicit missing-twin report.
    """
    try:
        importlib.import_module(_TWIN_PROVIDER_PACKAGE)
    except ModuleNotFoundError as exc:
        # Only swallow "the provider package is not installed"; a genuine import
        # error inside an existing package must surface.
        if exc.name != _TWIN_PROVIDER_PACKAGE:
            raise
        logger.warning("frontend.warp: twin provider package %r is not installed", _TWIN_PROVIDER_PACKAGE)


def _resolve_warp_twin(name: str, modules: Iterable[ModuleType]) -> Any | None:
    """Return the same-named symbol from ``modules`` that originates under the warp packages."""
    for module in modules:
        candidate = getattr(module, name, None)
        if candidate is None:
            continue
        origin = getattr(candidate, "__module__", "") or ""
        if origin.startswith(_WARP_ROOT_PREFIXES):
            return candidate
    return None


def _is_swap_candidate(value: Any) -> bool:
    """Heuristic: callable or class whose origin is *not already* warp."""
    if not callable(value):
        return False
    origin = getattr(value, "__module__", "") or ""
    if origin.startswith(_WARP_ROOT_PREFIXES):
        return False  # already a warp twin (idempotent)
    return True


def _walk_terms(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield ``(path, term)`` for every MDP term cfg in the cfg tree.

    A "term" is a :class:`ManagerTermBaseCfg` (observation/reward/termination/
    event/curriculum) *or* an :class:`ActionTermCfg` — the latter is a separate
    base that is **not** a ``ManagerTermBaseCfg`` subclass, yet carries a
    swappable ``class_type``, so it must be matched explicitly.

    Behavior at each node:

    * Match (a term cfg instance): yield ``(path, node)`` and stop — do not
      descend into ``term.params`` / ``term.func`` / ``term.class_type``.
    * Configclass: don't yield; recurse into every non-underscore *instance*
      attribute (``vars(node)``), extending the path. ``observations``,
      ``rewards``, ``events``, ``actions``, sub-groups like
      ``observations.policy`` / ``observations.perception``, and anything
      nested deeper are reached transparently.
    * Anything else (plain Python data, callables, non-configclass objects):
      stop. No yield, no recursion.

    Iterating the instance ``__dict__`` mirrors how the warp managers consume
    group cfgs (they iterate ``cfg.__dict__.items()``), so the walker sees
    exactly the terms the managers will see — including terms assigned in
    ``__post_init__`` — while never descending into methods or nested class
    objects, which live on the class rather than the instance.

    Driven entirely by type — no attribute names are hardcoded — so future
    cfg layouts (extra observation groups, new nesting, etc.) are picked up
    automatically as long as their terms subclass one of the term base cfgs.
    """
    from isaaclab.managers.manager_term_cfg import ActionTermCfg, ManagerTermBaseCfg

    if isinstance(node, (ManagerTermBaseCfg, ActionTermCfg)):
        yield path, node
        return
    if not hasattr(node, "__dataclass_fields__"):
        return
    for name, value in vars(node).items():
        if name.startswith("_") or value is None:
            continue
        yield from _walk_terms(value, path + (name,))
