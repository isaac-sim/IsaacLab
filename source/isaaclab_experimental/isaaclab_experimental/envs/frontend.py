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

# Stable task MDP packages and their Warp-first counterparts. The stable task
# package is organized by public task domain while the experimental package
# separates manager-based workflows by family, so the module trees are not
# one-to-one. Keep the routing at the package boundary and preserve any
# submodule suffix (for example, ``.rewards``).
_WARP_MDP_MODULE_ROUTES: tuple[tuple[str, str], ...] = (
    (
        "isaaclab_tasks.core.cartpole.mdp",
        "isaaclab_tasks_experimental.manager_based.classic.cartpole.mdp",
    ),
    (
        "isaaclab_tasks.core.locomotion.humanoid.mdp",
        "isaaclab_tasks_experimental.manager_based.classic.humanoid.mdp",
    ),
    (
        "isaaclab_tasks.core.velocity.mdp",
        "isaaclab_tasks_experimental.manager_based.locomotion.velocity.mdp",
    ),
    (
        "isaaclab_tasks.core.reach.mdp",
        "isaaclab_tasks_experimental.manager_based.manipulation.reach.mdp",
    ),
)

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
    callable. Twin lookup is name-based against the warp mirror of the *stable
    symbol's own module* (:func:`_warp_mdp_modules`) plus the
    :mod:`isaaclab_experimental.envs.mdp` fallback. Keying off the symbol's
    module — not the cfg's — means a task that borrows another task's MDP (e.g.
    ``manager_ant`` reuses ``manager_humanoid.mdp``) resolves to the right warp
    twins without a per-task shim, mirroring the stable ``core/`` layout. Any
    missing twin raises :class:`FrontendIncompatibleError` listing every
    affected term — partial swaps would leave torch-style callables in the cfg
    and the warp managers would call them with the wrong signature.

    The warp-side side declarations (``out_dim``, ``axes``, ``observation_type``)
    that the warp managers need at init are *not* supplied by this swap; they
    travel with the warp twin function itself via its own
    ``@generic_io_descriptor_warp(out_dim=…)`` decorator. This function only
    substitutes the callable; the manager reads the new func's annotations
    when it parses the term cfg.
    """
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
                module_cache[origin] = _warp_mdp_modules(origin)
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


def _warp_mdp_modules(symbol_module: str) -> list[ModuleType]:
    """Locate warp MDP modules to consult for a stable symbol's twin.

    The lookup is keyed off the *stable symbol's own module*
    (``func.__module__`` / ``class_type.__module__``), e.g.
    ``isaaclab_tasks.core.locomotion.humanoid.mdp.rewards``. The stable module
    is routed to its manager-based experimental counterpart, then walked up to
    the nearest importable module. Keying off the symbol's module — not the
    cfg's — resolves the right twins even when a task borrows another task's
    MDP (the Ant task reuses Humanoid MDP terms).

    Order of preference:

    1. The routed Warp counterpart of ``symbol_module`` (or its nearest importable ancestor,
       e.g. the ``...mdp`` package when the exact submodule isn't mirrored).
    2. The shared :mod:`isaaclab_experimental.envs.mdp` fallback (where
       generic warp twins live).
    """
    modules: list[ModuleType] = []
    warp_mod = _warp_mdp_module_name(symbol_module)
    if warp_mod is not None:
        parts = warp_mod.split(".")
        for depth in range(len(parts), 0, -1):
            target = ".".join(parts[:depth])
            try:
                modules.append(importlib.import_module(target))
                break
            except ModuleNotFoundError as exc:
                # Keep walking up while the missing module is part of the path
                # we're probing; a genuine import error inside an existing
                # module (missing third-party dep, etc.) must surface.
                if exc.name and warp_mod.startswith(exc.name):
                    continue
                raise
    # Generic fallback.
    fallback = "isaaclab_experimental.envs.mdp"
    try:
        modules.append(importlib.import_module(fallback))
    except ModuleNotFoundError as exc:
        if exc.name == fallback:
            logger.warning("frontend.warp: fallback mdp module %r not importable", fallback)
        else:
            raise
    return modules


def _warp_mdp_module_name(symbol_module: str) -> str | None:
    """Return the routed experimental MDP module for a stable task symbol."""
    if not isinstance(symbol_module, str):
        return None
    for stable_prefix, warp_prefix in _WARP_MDP_MODULE_ROUTES:
        if symbol_module == stable_prefix:
            return warp_prefix
        if symbol_module.startswith(f"{stable_prefix}."):
            return f"{warp_prefix}{symbol_module[len(stable_prefix) :]}"
    return None


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
    * Configclass: don't yield; recurse into every non-underscore attribute,
      extending the path. ``observations``, ``rewards``, ``events``, ``actions``,
      sub-groups like ``observations.policy`` / ``observations.perception``, and
      anything nested deeper are reached transparently.
    * Anything else (plain Python data, callables, non-configclass objects):
      stop. No yield, no recursion.

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
    for name in dir(node):
        if name.startswith("_"):
            continue
        try:
            value = getattr(node, name, None)
        except Exception:  # noqa: BLE001 — defensive; some descriptors can raise on attribute access
            continue
        if value is None:
            continue
        yield from _walk_terms(value, path + (name,))
