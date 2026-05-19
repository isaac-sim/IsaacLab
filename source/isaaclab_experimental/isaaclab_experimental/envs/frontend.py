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

# Term containers walked by the cfg-adaptation step. Actions are included so
# the swap logic handles ``class_type`` and ``func`` in one pass.
_TERM_PATHS: tuple[tuple[str, ...], ...] = (
    ("observations", "policy"),
    ("events",),
    ("rewards",),
    ("terminations",),
    ("commands",),
    ("curriculum",),
    ("actions",),
)


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
        # under the warp packages (e.g. ``Isaac-Cartpole-Direct-Warp-v0``).
        _require_direct_is_warp_task(task_id)
        return gym.make(task_id, cfg=env_cfg, **construct_kwargs)

    _adapt_cfg_for_warp(env_cfg, task_id)
    # Imported lazily so that ``--frontend=torch`` callers don't pay the
    # ``isaaclab_experimental.envs`` import cost.
    from isaaclab_experimental.envs import ManagerBasedRLEnvWarp

    return ManagerBasedRLEnvWarp(cfg=env_cfg, **construct_kwargs)


# ---------------------------------------------------------------------------
# Cfg adaptation (warp only)
# ---------------------------------------------------------------------------


def _adapt_cfg_for_warp(cfg: Any, task_id: str) -> None:
    """Mutate ``cfg`` in place so warp managers can consume it.

    Three steps, each independently testable:

    1. :func:`_require_newton_physics` — hard check that ``cfg.sim.physics`` is
       :class:`~isaaclab_newton.physics.NewtonCfg`. The user is responsible for
       selecting the Newton variant of the task's :class:`PresetCfg` via
       ``presets=newton``; we don't auto-inject.
    2. :func:`_promote_scene_entity_cfgs` — replace stable
       :class:`~isaaclab.managers.SceneEntityCfg` instances under each term's
       ``params`` with the warp variant (which adds warp-cached ``joint_mask``,
       ``joint_ids_wp``, ``body_ids_wp`` fields).
    3. :func:`_swap_mdp` — for every MDP term in every group (observations,
       events, rewards, terminations, commands, curriculum, actions), replace
       any stable ``func`` *or* ``class_type`` with its same-named warp twin.
       A missing twin raises :class:`FrontendIncompatibleError` — partial
       coverage is unsafe under the warp managers' kernel-only signature.
    """
    _require_newton_physics(cfg, task_id)
    _promote_scene_entity_cfgs(cfg)
    _swap_mdp(cfg, task_id)


def _require_newton_physics(cfg: Any, task_id: str) -> None:
    """Block unless ``cfg.sim.physics`` is :class:`NewtonCfg`.

    The warp managers' assets read state through :class:`NewtonManager`;
    a :class:`PhysxCfg` (or unresolved :class:`PresetCfg`) is a hard
    incompatibility. The fix is to pass ``presets=newton`` on the CLI so
    Hydra resolves the task's :class:`PresetCfg` wrapper to the Newton field
    before construction.
    """
    from isaaclab_newton.physics import NewtonCfg

    physics = getattr(getattr(cfg, "sim", None), "physics", None)
    if isinstance(physics, NewtonCfg):
        return
    raise FrontendIncompatibleError(
        f"--frontend=warp on {task_id!r}: expected cfg.sim.physics to be NewtonCfg,"
        f" got {type(physics).__name__!r}. Pass `presets=newton` on the CLI so"
        f" Hydra resolves the task's PresetCfg wrapper to the Newton variant."
    )


def _promote_scene_entity_cfgs(cfg: Any) -> None:
    """Replace stable :class:`SceneEntityCfg` instances with the warp variant.

    Walks every ``term.params: dict`` under each term group and rebuilds any
    stable :class:`SceneEntityCfg` value via
    :meth:`isaaclab_experimental.managers.SceneEntityCfg.from_stable`. The
    warp variant subclasses the stable one, so type checks elsewhere stay
    valid; the new fields (``joint_mask`` / ``joint_ids_wp`` / ``body_ids_wp``)
    are filled at :meth:`resolve` time by the warp scene.
    """
    from isaaclab_experimental.managers.scene_entity_cfg import SceneEntityCfg as _WarpSceneEntityCfg

    promoted = 0
    for path in _TERM_PATHS:
        group = _walk_attrs(cfg, path)
        if group is None:
            continue
        for _name, term in _iter_term_attrs(group):
            params = getattr(term, "params", None)
            if not isinstance(params, dict):
                continue
            for key, value in list(params.items()):
                if isinstance(value, _WarpSceneEntityCfg) or not isinstance(value, _StableSceneEntityCfg):
                    continue
                params[key] = _WarpSceneEntityCfg.from_stable(value)
                promoted += 1
    if promoted:
        logger.info("frontend.warp: promoted %d SceneEntityCfg instance(s) to warp variant", promoted)


def _swap_mdp(cfg: Any, task_id: str) -> None:
    """Replace ``term.func`` and ``term.class_type`` with their warp twins.

    Walks the same term paths used by :func:`_promote_scene_entity_cfgs` and
    on each term swaps whichever of ``func`` / ``class_type`` is set to a
    stable-origin symbol. Twin lookup is name-based against the task's
    matching ``isaaclab_tasks_experimental.<...>.mdp`` module and, as a
    fallback, :mod:`isaaclab_experimental.envs.mdp`. Any missing twin raises
    :class:`FrontendIncompatibleError` listing every affected term — partial
    swaps would leave torch-style callables in the cfg and the warp managers
    would call them with the wrong signature.
    """
    modules = _warp_mdp_modules(task_id)
    searched = tuple(m.__name__ for m in modules)
    logger.info("frontend.warp: searching warp mdp modules %s", list(searched))

    swapped = 0
    missing: list[tuple[str, str, str]] = []  # (location, attr, symbol)
    for path in _TERM_PATHS:
        group = _walk_attrs(cfg, path)
        if group is None:
            continue
        location_prefix = ".".join(path)
        for name, term in _iter_term_attrs(group):
            for attr in ("func", "class_type"):
                stable = getattr(term, attr, None)
                if stable is None or not _is_swap_candidate(stable):
                    continue
                twin = _resolve_warp_twin(stable.__name__, modules)
                if twin is None:
                    missing.append((f"{location_prefix}.{name}", attr, stable.__name__))
                    continue
                setattr(term, attr, twin)
                swapped += 1

    if missing:
        lines = "\n  ".join(f"{loc}.{attr}: no warp twin for {sym!r}" for loc, attr, sym in missing)
        raise FrontendIncompatibleError(
            f"--frontend=warp on {task_id!r}: missing warp MDP twins (searched {list(searched)}):\n  {lines}"
        )

    logger.info("frontend.warp: swapped %d MDP symbol(s) to warp twins", swapped)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_workflow(cfg: Any) -> Workflow:
    """Classify the env cfg into manager-based or direct (used to pick build path)."""
    if isinstance(cfg, ManagerBasedRLEnvCfg):
        return Workflow.MANAGER_BASED
    if isinstance(cfg, (DirectRLEnvCfg, DirectMARLEnvCfg)):
        return Workflow.DIRECT
    raise FrontendIncompatibleError(
        f"Unrecognised env cfg type {type(cfg).__name__!r};"
        f" expected ManagerBasedRLEnvCfg / DirectRLEnvCfg / DirectMARLEnvCfg subclass."
    )


def _require_direct_is_warp_task(task_id: str) -> None:
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


def _warp_mdp_modules(task_id: str) -> list[ModuleType]:
    """Locate warp MDP modules to consult for twin lookups.

    Order of preference:

    1. The task-specific module, derived by replacing ``isaaclab_tasks`` with
       ``isaaclab_tasks_experimental`` in the task's ``env_cfg_entry_point``
       package and walking up to the first existing ``.mdp`` submodule.
    2. The shared :mod:`isaaclab_experimental.envs.mdp` fallback (where
       generic warp twins live).
    """
    modules: list[ModuleType] = []
    try:
        spec = gym.spec(task_id)
    except gym.error.NameNotFound:
        spec = None
    entry = spec.kwargs.get("env_cfg_entry_point") if spec is not None else None
    if isinstance(entry, str) and entry.startswith("isaaclab_tasks."):
        warp_pkg = entry.rsplit(".", 1)[0].replace("isaaclab_tasks", "isaaclab_tasks_experimental", 1)
        parts = warp_pkg.split(".")
        for depth in range(len(parts), 0, -1):
            target = ".".join(parts[:depth] + ["mdp"])
            try:
                modules.append(importlib.import_module(target))
                break
            except ModuleNotFoundError as exc:
                # Only swallow "this candidate doesn't exist" — a real
                # import error inside an existing module is a bug.
                if exc.name == target:
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


def _walk_attrs(root: Any, path: tuple[str, ...]) -> Any:
    """Walk ``root.<path[0]>.<path[1]>…``; return ``None`` on any miss."""
    node = root
    for attr in path:
        node = getattr(node, attr, None)
        if node is None:
            return None
    return node


def _iter_term_attrs(group: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, term)`` pairs from a manager-cfg group, skipping dunders/Nones."""
    if group is None:
        return
    for name in [n for n in dir(group) if not n.startswith("_")]:
        term = getattr(group, name, None)
        if term is None:
            continue
        yield name, term
