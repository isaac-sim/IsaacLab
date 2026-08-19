# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Enumerate registered training tasks and the backend combinations they support.

Declared combinations come from one registry walk. Resolved ones cost a config build
and a validator run each, and are the only reliable answer: the cross product is not
all legal, since OVRTX cannot share a process with Kit physics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RL_LIBRARY_PRIORITY",
    "is_training_task",
    "DiscoveredTask",
    "DiscoveryError",
    "discover_tasks",
]

# Shared with the environment tables, but only the ordering: ``RL_LIBRARY_OVERRIDES``
# there adds libraries declaring no entry point, so ``rlinf`` can appear in the tables
# and not in :attr:`DiscoveredTask.rl_libraries`.
RL_LIBRARY_PRIORITY: tuple[str, ...] = ("rl_games", "rsl_rl", "skrl", "sb3", "rlinf")

# ``-Eval`` variants are separate registrations against an eval-specific env cfg, and
# ``-Benchmark-`` tasks are perf harnesses; neither is a training row.
_EVAL_TASK_SUFFIXES = ("-Eval",)

# The validator could not run, as opposed to rejecting the combination. Swallowing
# these would mark every combination illegal. ``ImportError`` counts as drift because a
# missing *extra* raises ``ModuleNotFoundError`` and is handled before them.
_INFRASTRUCTURE_ERRORS = (ImportError, AttributeError, NameError, SyntaxError, TypeError)


class DiscoveryError(RuntimeError):
    """Raised when the registry cannot be walked, or validation cannot run."""


@dataclass(frozen=True)
class DiscoveredTask:
    """One registered training task.

    Args:
        task_id: Gym task id.
        scope: ``core`` or ``contrib``.
        rl_libraries: Libraries the task declares an agent config for, in
            :data:`RL_LIBRARY_PRIORITY` order. Empty for IK, teleop and mimic tasks.
        declared: Preset names by axis, or ``None`` if the config would not load. An
            all-empty mapping means the task declares none. Keeps duplicate spellings.
        modes: Ways to run the task; see :func:`_build_modes`. Resolution assumes a
            headless launch, reads ``LIVESTREAM`` from the environment, and stops before
            the Isaac Sim availability check.
        default: The no-token run. ``None`` in declared mode, or if it does not resolve.
        resolved: Whether ``modes`` was resolved or merely declared.
    """

    @dataclass(frozen=True)
    class Mode:
        """The tokens one run passes, ``None`` on an axis it leaves to the config.

        ``presets`` holds at most one name even though the token takes a comma list, so
        a task with several independent preset axes is under-approximated.
        """

        physics: str | None
        renderer: str | None
        presets: str | None

    @dataclass(frozen=True)
    class Default:
        """The no-token run: the ``modes`` entry it collapsed into, and the physics
        cfg class it resolves to, e.g. ``NewtonCfg(MJWarpSolverCfg)``.
        """

        backend: str | None
        mode: DiscoveredTask.Mode

    task_id: str
    scope: str
    rl_libraries: tuple[str, ...]
    # Mutable, so instances are not hashable despite ``frozen``. Treat as read-only.
    declared: dict[str, tuple[str, ...]] | None
    modes: tuple[DiscoveredTask.Mode, ...]
    default: DiscoveredTask.Default | None
    resolved: bool


def _domain_presets(names: list[str], typed_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return domain presets, dropping those the task also declares on a typed axis.

    Filtering by name instead would hide backends only reachable as ``presets=NAME``.
    """
    typed = set(typed_names)
    return tuple(sorted(name for name in names if name not in typed))


def is_training_task(task_id: str) -> bool:
    """Return whether *task_id* is a trainable (non-inference) Isaac task."""
    if "Isaac" not in task_id:
        return False
    if any(task_id.endswith(suffix) for suffix in _EVAL_TASK_SUFFIXES):
        return False
    return "-Benchmark-" not in task_id


def _rl_libraries_from_kwargs(kwargs: dict[str, Any]) -> tuple[str, ...]:
    """Return the RL libraries a registration declares an agent config for.

    Matched on the stem before ``_cfg_entry_point`` so variants such as
    ``rsl_rl_recurrent_cfg_entry_point`` count towards their library.
    """
    declared = set()
    for key in kwargs:
        if not key.endswith("_cfg_entry_point") or key == "env_cfg_entry_point":
            continue
        stem = key[: -len("_cfg_entry_point")]
        for candidate in RL_LIBRARY_PRIORITY:
            if stem == candidate or stem.startswith(f"{candidate}_"):
                declared.add(candidate)
                break
    return tuple(name for name in RL_LIBRARY_PRIORITY if name in declared)


def _mode_resolves(
    task_id: str, physics: str | None, renderer: str | None, presets: str | None = None
) -> tuple[str, str | None] | None:
    """Resolve one combination.

    Returns:
        ``None`` if it cannot run. Otherwise ``(fingerprint, backend)``: the fingerprint
        digests the resolved config, so two spellings of one run share it, and the
        backend carries the solver that separates ``newton_mjwarp`` from
        ``newton_kamino``.

    Raises:
        DiscoveryError: If validation could not run at all.
    """
    import argparse
    import hashlib
    import sys

    # Two are private. Losing one is drift, not a rejected combination.
    try:
        from isaaclab.app.sim_launcher import _get_kit_runtime_sources, _validate_runtime, scan

        from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
    except ImportError as exc:
        raise DiscoveryError(f"discovery could not import the Isaac Lab APIs it depends on: {exc}") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--agent", default=None)
    argv = ["--task", task_id]
    if physics is not None:
        argv.append(f"physics={physics}")
    if renderer is not None:
        argv.append(f"renderer={renderer}")
    if presets is not None:
        argv.append(f"presets={presets}")

    original_argv = list(sys.argv)
    try:
        args, remaining = setup_preset_cli(parser, argv)
        sys.argv = [sys.argv[0]] + remaining
        env_cfg, _ = resolve_task_config(args.task, args.agent)
        config_scan = scan(env_cfg, args)
        _validate_runtime(config_scan, _get_kit_runtime_sources(config_scan, args))
        fingerprint = hashlib.sha256(repr(env_cfg.to_dict()).encode()).hexdigest()
        physics_cfg = config_scan.resolved_physics_cfg
        solver = getattr(physics_cfg, "solver_cfg", None)
        backend = None if physics_cfg is None else type(physics_cfg).__name__
        if solver is not None:
            backend = f"{backend}({type(solver).__name__})"
        return fingerprint, backend
    except ModuleNotFoundError:
        # An uninstalled extra. Narrower than ``ImportError``, which would hide drift.
        return None
    except _INFRASTRUCTURE_ERRORS as exc:
        raise DiscoveryError(
            f"preset validation for {task_id!r} could not run: {type(exc).__name__}: {exc}. This is an Isaac Lab"
            " API failure, not a rejected preset combination."
        ) from exc
    except Exception:  # noqa: BLE001 - any other failure means the combination cannot run
        return None
    finally:
        sys.argv = original_argv


def _build_modes(
    task_id: str,
    physics: tuple[str, ...],
    renderers: tuple[str, ...],
    domains: tuple[str, ...],
    *,
    resolve: bool,
    strict: bool = False,
    collapse: bool = True,
) -> tuple[tuple[DiscoveredTask.Mode, ...], DiscoveredTask.Default | None]:
    """Return the runs for one task, and what it does when given no tokens.

    Renderers expand across, domain presets one at a time. ``collapse`` deduplicates on
    the resolved config, not the backend, which would merge same-backend runs.

    Returns:
        ``(modes, default)``. *default* names an explicit spelling even when ``collapse``
        is off, and is ``None`` in declared mode or when the no-token run cannot resolve.

    Raises:
        DiscoveryError: If ``strict`` and the validator could not judge a combination.
    """
    physics_options: tuple[str | None, ...] = physics or (None,)
    renderer_options: tuple[str | None, ...] = renderers or (None,)
    domain_options: tuple[str | None, ...] = (None, *domains) if domains else (None,)

    if not resolve:
        return (
            tuple(
                DiscoveredTask.Mode(physics=p, renderer=r, presets=d)
                for p in physics_options
                for r in renderer_options
                for d in domain_options
            ),
            None,
        )

    combinations = [(p, r, d) for p in physics_options for r in renderer_options for d in domain_options]
    # The no-token run always has to be probed, and is not always in the cross product:
    # a task declaring physics presets has no ``physics=None`` column.
    if (None, None, None) not in combinations:
        combinations.insert(0, (None, None, None))

    # Keyed by fingerprint, holding the most explicit spelling seen of each run -- naming
    # the presets reproduces it even if the config's own defaults move later. Built even
    # when uncollapsed, since it identifies the spelling of the no-token run.
    unique: dict[str, tuple[int, DiscoveredTask.Mode]] = {}
    validated: list[DiscoveredTask.Mode] = []
    default_key: str | None = None
    default_backend: str | None = None
    for physics_name, renderer, domain in combinations:
        try:
            resolution = _mode_resolves(task_id, physics_name, renderer, domain)
        except DiscoveryError as exc:
            if strict:
                raise
            # Unknown is not the same answer as legal, so drop it -- but loudly,
            # because it is a gap in the matrix.
            logger.warning(
                "%s: dropping physics=%s renderer=%s presets=%s: %s", task_id, physics_name, renderer, domain, exc
            )
            continue
        if resolution is None:
            continue
        key, backend = resolution
        mode = DiscoveredTask.Mode(physics=physics_name, renderer=renderer, presets=domain)
        validated.append(mode)
        if (physics_name, renderer, domain) == (None, None, None):
            default_key, default_backend = key, backend
        explicitness = sum(token is not None for token in (physics_name, renderer, domain))
        incumbent = unique.get(key)
        if incumbent is None or explicitness > incumbent[0]:
            unique[key] = (explicitness, mode)

    default = None
    if default_key is not None:
        default = DiscoveredTask.Default(backend=default_backend, mode=unique[default_key][1])
    modes = tuple(mode for _, mode in unique.values()) if collapse else tuple(validated)
    return modes, default


def discover_tasks(
    specs: list[Any] | None = None, *, resolve: bool = True, strict: bool = False, collapse: bool = True
) -> list[DiscoveredTask]:
    """Walk the Gym registry and return every registered training task.

    Imports Isaac Lab, so it needs the project environment. ``isaaclab_tasks`` registers
    the core and contrib tasks; ``isaaclab_tasks_experimental`` is imported too when
    present, for whatever it registers.

    Args:
        specs: Gym specs to walk. When ``None``, the whole registry is scanned.
        resolve: Build every combination and keep only what the validator accepts.
            When ``False``, report what is declared: fast, unverified.
        strict: Raise on a combination the validator cannot judge instead of logging
            and dropping it. Off by default, so the task is still returned with that
            combination missing from ``modes``.
        collapse: Reduce spellings resolving to the same config to one. Off keeps every
            validated spelling. Ignored when ``resolve`` is off.

    Returns:
        Discovered tasks sorted by ``task_id``.

    Raises:
        DiscoveryError: If the task packages cannot be imported, or, when ``strict``, if
            any task could not be inspected.
    """
    import contextlib

    try:
        import gymnasium as gym

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
        from isaaclab_tasks.utils.preset_target import PresetTarget
    except ImportError as exc:
        raise DiscoveryError(f"could not import the Isaac Lab task packages: {exc}") from exc

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    if specs is None:
        specs = list(gym.registry.values())

    tasks: list[DiscoveredTask] = []
    for spec in specs:
        if not is_training_task(spec.id) or spec.kwargs.get("deprecated"):
            continue
        # Tasks with no RL entry point (IK, teleop, mimic) are still registered
        # environments; callers needing a trainable task filter on ``rl_libraries``.
        preset_map = enumerate_task_presets(spec.id)
        declared_physics = tuple(sorted(preset_map.get(PresetTarget.PHYSICS, []))) if preset_map else ()
        renderers = tuple(sorted(preset_map.get(PresetTarget.RENDERER, []))) if preset_map else ()
        domains = (
            _domain_presets(preset_map.get(PresetTarget.DOMAIN, []), declared_physics + renderers) if preset_map else ()
        )
        if preset_map is None:
            # Nothing is known about this task, and a cross product of nothing known is
            # not a cross product of nothing declared -- do not invent a runnable mode.
            modes, default = (), None
        else:
            modes, default = _build_modes(
                spec.id, declared_physics, renderers, domains, resolve=resolve, strict=strict, collapse=collapse
            )
        tasks.append(
            DiscoveredTask(
                task_id=spec.id,
                scope="contrib" if spec.id.startswith("IsaacContrib-") else "core",
                rl_libraries=_rl_libraries_from_kwargs(spec.kwargs),
                declared=(
                    None
                    if preset_map is None
                    else {
                        "physics": declared_physics,
                        "renderer": renderers,
                        "presets": tuple(sorted(preset_map.get(PresetTarget.DOMAIN, []))),
                    }
                ),
                modes=modes,
                default=default,
                resolved=resolve,
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    return tasks
