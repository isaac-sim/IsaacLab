# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Enumerate registered training tasks and the backend combinations they support.

Two questions get asked of the Gym registry, and they have different answers:

* What does a task **declare**? One registry walk. Fast, and unverified.
* What does a task actually **resolve**? One config build and validator run per
  combination -- minutes for the full registry, but the only way to know a
  combination can run. The cross product is not all legal: OVRTX is kitless and
  cannot share a process with Kit physics, so ``isaacsim_physx + ovrtx`` is
  declared yet unusable.

:func:`discover_tasks` answers either, selected with ``resolve``.

Resolving also identifies each run by the config it produces rather than by how it
was spelled, so spellings that produce the same run collapse: ``physics=physx``
folds into the concrete backend it resolves to, and passing nothing folds into
whichever preset the config already defaults to. That last run is reported
separately as ``default``, which the collapse would otherwise hide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RL_LIBRARY_PRIORITY",
    "is_training_task",
    "DiscoveredTask",
    "DiscoveryError",
    "discover_tasks",
]

# Stable ordering for the RL library axis, shared with the environment tables so the
# two never disagree. ``rlinf`` has no discoverable entry point but is listed for
# ordering, since :data:`~environ_docs.RL_LIBRARY_OVERRIDES` supplies it.
RL_LIBRARY_PRIORITY: tuple[str, ...] = ("rl_games", "rsl_rl", "skrl", "sb3", "rlinf")

# ``-Eval`` marks dedicated evaluation variants registered as aliases, which should
# not appear as their own training row.
_EVAL_TASK_SUFFIXES = ("-Eval",)

# Errors meaning the validator itself could not run, rather than that the combination
# under test was rejected. Swallowing them would mark every combination illegal, so
# they are logged and the combination dropped; ``strict=True`` re-raises instead.
# ``TypeError`` is included because calling the validator with the wrong argument type
# is otherwise indistinguishable from a rejected combination.
_INFRASTRUCTURE_ERRORS = (ImportError, AttributeError, NameError, SyntaxError, TypeError)


class DiscoveryError(RuntimeError):
    """Raised when the registry cannot be walked, or validation cannot run."""


@dataclass(frozen=True)
class DiscoveredTask:
    """One registered training task.

    Args:
        task_id: Gym task id.
        scope: ``core`` or ``contrib``.
        rl_libraries: RL libraries the task declares an agent config for, in
            :data:`RL_LIBRARY_PRIORITY` order. Empty for registered environments with
            no RL entry point, such as IK, teleop and mimic tasks.
        declared: Preset names the task declares, keyed by axis (``physics``,
            ``renderer``, ``presets``) exactly as the registry reports them, or ``None``
            when the config could not be loaded. ``None`` means unknown; an all-empty
            mapping means the task declares nothing and runs on a fixed backend.
            Unreconciled: aliases and cross-axis duplicates are all present. Use
            ``modes`` for a deduplicated answer.
        modes: Ways to run the task. In resolved mode these passed the validator, and
            with ``collapse`` are reduced so each is a distinct run rather than a
            distinct spelling; without it every validated spelling is kept. In declared
            mode they are the raw cross product, unverified.

            Validation assumes a **headless launch** -- no ``--visualizer``,
            ``--livestream``, ``--experience`` or ``--require_kit``. Each is a Kit
            source, so adding one narrows the legal set.
        default: What the task does when given no preset tokens, or ``None`` in
            declared mode.
        resolved: Whether ``modes`` was resolved, or merely declared.
    """

    @dataclass(frozen=True)
    class Mode:
        """One way to run a task.

        Args:
            physics: Physics preset token, or ``None`` for tasks declaring none.
            renderer: Renderer preset token, or ``None`` to run headless.
            presets: Domain preset token passed as ``presets=<value>``, or ``None``.
                Never more than one. ``presets=`` does take a comma-separated list and
                names on different config paths compose (``duo_camera,depth128,cube``);
                only names sharing a path conflict (``depth,rgb``). Discovery validates
                each name alone and never tries a pair, so ``modes`` under-approximates
                a task with several independent preset axes.
        """

        physics: str | None
        renderer: str | None
        presets: str | None

    @dataclass(frozen=True)
    class Default:
        """The run a task performs when given no preset tokens.

        Args:
            backend: Concrete physics config the run resolves to, e.g. ``PhysxCfg`` or
                ``NewtonCfg(MJWarpSolverCfg)``. Reported as the config class because a
                task's default need not have a preset name.
            mode: The entry in ``modes`` this run collapsed into.
        """

        backend: str | None
        mode: DiscoveredTask.Mode

    task_id: str
    scope: str
    rl_libraries: tuple[str, ...]
    declared: dict[str, tuple[str, ...]] | None = field(default_factory=dict)
    modes: tuple[DiscoveredTask.Mode, ...] = ()
    default: DiscoveredTask.Default | None = None
    resolved: bool = False


def _domain_presets(names: list[str], typed_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return domain presets, dropping those the task also declares on a typed axis.

    A backend buckets under ``DOMAIN`` or under ``PHYSICS`` / ``RENDERER`` depending on
    whether its cfg class subclasses ``PhysicsCfg`` / ``RendererCfg``, so the same name
    means different things on different tasks. Declared on a typed axis too, it is
    reachable as ``physics=NAME`` and reporting it again double-counts one run; declared
    only here, ``presets=NAME`` is the sole way to select it and ``physics=NAME`` is
    rejected. Deciding by name rather than per task hides every backend of the latter.
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
    """Resolve one combination and identify the run it produces.

    Returns:
        ``None`` when the combination cannot run -- an unknown preset, an unloadable
        config and a rejected backend pairing are the same answer. Otherwise
        ``(fingerprint, backend)``, where *fingerprint* digests the resolved env config
        so two spellings of one run share it, and *backend* names the concrete physics
        config, e.g. ``NewtonCfg(MJWarpSolverCfg)``. The solver is included because it
        is what separates ``newton_mjwarp`` from ``newton_kamino``.

    Raises:
        DiscoveryError: If validation could not run at all, e.g. because an Isaac Lab
            import or API it depends on has changed.
    """
    import argparse
    import hashlib
    import sys

    from isaaclab.app.sim_launcher import _get_kit_runtime_sources, _validate_runtime, scan

    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

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
    except ImportError:
        # The config needs an extra that is not installed, so the combination cannot
        # run here. Same answer as a rejection, and it keeps a partial install usable.
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

    Renderers are expanded across, since reporting a camera task as headless-only omits
    the thing under test. Domain presets are expanded one at a time, never combined.

    ``collapse`` deduplicates on the resolved config, so each mode is a distinct run.
    Collapsing on the *backend* instead would be wrong -- the Reach controller presets
    share a backend and are four different runs. Without it every validated spelling is
    kept, which is what documentation needs: a preset naming the default of its own axis
    is still a token a reader can type.

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

    Imports Isaac Lab, so it needs the project environment. Contrib tasks are included
    when ``isaaclab_tasks_experimental`` is importable.

    Args:
        specs: Gym specs to walk. When ``None``, the whole registry is scanned.
        resolve: When ``True``, every combination is built and checked against the
            runtime validator and only usable ones are returned. When ``False``,
            combinations are reported as declared: fast but unverified.
        strict: When ``True``, a task the validator cannot run against raises instead of
            being logged and skipped. Off by default so one broken task costs the caller
            that task rather than the whole registry; turn it on to police API drift.
        collapse: When ``True`` (the default), spellings resolving to the same config are
            reduced to one, so ``modes`` is the distinct runs a dispatcher should
            schedule. Turn it off to keep every validated spelling, which is what
            documentation needs. Ignored when ``resolve`` is off.

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
