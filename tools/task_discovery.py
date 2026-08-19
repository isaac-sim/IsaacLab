# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Enumerate registered training tasks and the backend combinations they support.

Two questions get asked of the Gym registry, and they do not have the same answer:

* What does a task **declare**? Reading :func:`~isaaclab_tasks.utils.preset_cli.enumerate_task_presets`
  is fast and is what the environment documentation reports.
* What does a task actually **resolve**? Building the config and running the runtime
  validator is slow, but it is the only way to know a combination can run. The cross
  product is not all legal: OVRTX is kitless and cannot share a process with Kit
  physics, so ``isaacsim_physx + ovrtx`` is declared yet unusable.

:func:`discover_tasks` answers either, selected with ``resolve``. Declared mode costs
one registry walk; resolved mode additionally costs one config resolution per
combination, which is minutes for the full registry but far cheaper than finding out
on a GPU.

Resolving buys more than a legality check. Because each combination is resolved all
the way to an env config, combinations can be compared on what they *produce* rather
than on how they were spelled, and the ones that produce the same run collapse. That
is what makes ``modes`` a list of distinct runs: ``physics=physx`` folds into whatever
concrete backend it resolves to, and passing no tokens at all folds into whichever
preset the config already defaults to. Aliases need no table of names, and a
dispatcher can run every mode without repeating work. What the task does when given
no tokens is kept separately as ``default`` — the collapse would otherwise hide it.

The gap between declared and resolved is itself useful: a combination that is declared
but does not resolve is documentation drift.
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

# Gym IDs excluded from the training list. The ``-Eval`` suffix marks dedicated
# evaluation variants (e.g. ``IsaacContrib-Assemble-Trocar-G129-Dex3-Eval``, an alias
# registered for RLinf eval configs) that should not appear as their own training row.
_EVAL_TASK_SUFFIXES = ("-Eval",)

# Errors that mean the validator itself could not run, rather than that the
# combination under test was rejected. They are logged and the combination is
# dropped, so one broken task costs the caller that task and not the whole walk;
# ``strict=True`` re-raises instead, for callers policing Isaac Lab API drift.
# ``TypeError`` is included because calling the validator with the wrong argument
# type is otherwise indistinguishable from a rejected combination.
_INFRASTRUCTURE_ERRORS = (ImportError, AttributeError, NameError, SyntaxError, TypeError)


class DiscoveryError(RuntimeError):
    """Raised when the registry cannot be walked, or validation cannot run."""


@dataclass(frozen=True)
class DiscoveredTask:
    """One registered training task.

    Args:
        task_id: Gym task id.
        scope: ``core`` or ``contrib``.
        rl_libraries: RL libraries the task declares, in :data:`RL_LIBRARY_PRIORITY`
            order. Empty for registered environments with no RL entry point, such as
            IK, teleop and mimic tasks.
        declared: Preset names the task declares, keyed by axis (``physics``,
            ``renderer``, ``presets``), exactly as the registry reports them, or
            ``None`` when the config could not be loaded at all. ``None`` and an
            all-empty mapping are different answers: the first means unknown, the
            second means the task declares nothing and runs on a fixed backend.
            Nothing is filtered out: a backend a task exposes on both a typed axis
            and as a ``presets=`` token appears under both, and aliases such as
            ``physx`` are present alongside what they resolve to. Use ``modes`` for
            a deduplicated answer; ``declared`` is the unreconciled registry view.
        modes: Ways to run the task. In resolved mode these passed the runtime
            validator; with ``collapse`` they are further reduced so that two token
            spellings producing the same resolved config appear once — no aliases, no
            double-counting. Without it every validated spelling is kept, which is what
            a table of "what can I pass?" needs. In declared mode they are the raw
            cross product, unverified and uncollapsed.

            Validation assumes a **headless launch**: no ``--visualizer``, no
            ``--livestream``, no ``--experience`` and no ``--require_kit``. Each of
            those is a Kit source, so adding one narrows the legal set — a kitless
            OvPhysX combination that passes here is rejected under ``--visualizer kit``.
        default: What the task does when the user passes no preset tokens, or ``None``
            in declared mode. Reported separately because the collapse folds the
            no-token run into whichever named mode it matches, and a table still wants
            to say what you get if you change nothing.
        resolved: Whether ``modes`` was resolved and collapsed, or merely declared.
    """

    @dataclass(frozen=True)
    class Mode:
        """One way to run a task.

        Args:
            physics: Physics preset token, or ``None`` for tasks that declare none
                and reject any ``physics=`` selector.
            renderer: Renderer preset token, or ``None`` to run headless.
            presets: Domain preset token passed as ``presets=<value>``, or ``None``.
                Never more than one. ``presets=`` does accept a comma-separated list,
                and names on *different* config paths compose fine — on
                ``Isaac-Lift-KukaAllegro-Camera``, ``presets=duo_camera,depth128,cube``
                sets the camera count, the modality and the object independently. Only
                names sharing a path conflict, e.g. ``presets=depth,rgb``. Discovery
                validates each name on its own and never tries a pair, so ``modes``
                under-approximates a task with several independent preset axes.
        """

        physics: str | None
        renderer: str | None
        presets: str | None

    @dataclass(frozen=True)
    class Default:
        """The run a task performs when given no preset tokens.

        Args:
            backend: Concrete physics config the run resolves to, e.g. ``PhysxCfg``
                or ``NewtonCfg(MJWarpSolverCfg)``. Reported as the config class rather
                than a preset name because a task's default need not have one.
            mode: The entry in ``modes`` this run collapsed into — the explicit way to
                ask for the same thing.
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
    """Return domain presets, dropping the ones that mirror a typed selector.

    Whether a backend lands under ``PresetTarget.DOMAIN`` or under ``PHYSICS`` /
    ``RENDERER`` depends on whether its cfg class subclasses ``PhysicsCfg`` /
    ``RendererCfg``, so the same name means different things on different tasks:

    * Also declared on a typed axis — reachable as ``physics=NAME``, so reporting it
      again as ``presets=NAME`` double-counts one run. Dropped.
    * Not declared on a typed axis — ``presets=NAME`` is the *only* way to select
      that backend, and ``physics=NAME`` is rejected outright. Kept.

    Deciding by name instead of per task gets the second case backwards and silently
    hides every backend such a task has (``Isaac-Open-Drawer-Franka`` has five).

    A backend name surviving here can also mean the task is inconsistent. Shared
    configs pair a backend with *companion* overrides under the same name --
    ``velocity_env_cfg`` sets ``events.base_com=None`` under ``newton_mjwarp``,
    because Newton does not support that randomization. The companion is normally
    invisible: it rides along with the ``newton_mjwarp`` already on the physics axis
    and is dropped as a mirror. It only shows up as a standalone ``presets=`` token
    on a task that inherited the companion without offering the backend, where
    selecting it applies a Newton workaround to a PhysX run. Reporting it is correct
    -- the token is real and does change the config -- and it is worth reading as a
    signal to fix the task.
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

    Entry points are matched on the stem before ``_cfg_entry_point`` so that
    variants such as ``rsl_rl_recurrent_cfg_entry_point`` count towards their
    library rather than being dropped.
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
    """Resolve one physics/renderer/preset combination and identify the run it produces.

    An unknown preset, an unloadable config, or a rejected backend combination all
    mean the same thing — the combination cannot run — so they return ``None`` alike.

    Returns:
        ``None`` when the combination cannot run, else ``(fingerprint, backend)``.
        *fingerprint* digests the fully resolved env config, so two token spellings
        that produce the same run share it — ``presets=physx`` and ``presets=ovphysx``
        on the cabinet tasks, or passing nothing at all and naming the preset the
        config already defaults to. *backend* names the concrete physics config the
        run ends up with, e.g. ``PhysxCfg`` or ``NewtonCfg(MJWarpSolverCfg)``.

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
        # ``_validate_runtime`` takes the resolved Kit sources, not the parsed args.
        # Passing args makes every scan look Kit-backed, which fires the OvPhysX
        # guard for every OvPhysX combination and marks them all unusable.
        config_scan = scan(env_cfg, args)
        _validate_runtime(config_scan, _get_kit_runtime_sources(config_scan, args))
        fingerprint = hashlib.sha256(repr(env_cfg.to_dict()).encode()).hexdigest()
        return fingerprint, _backend_name(config_scan.resolved_physics_cfg)
    except ImportError:
        # The task's config needs an extra that is not installed, so this
        # combination cannot run in this environment. That is the same answer as a
        # rejected combination, and it keeps discovery usable from a partial install.
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


def _backend_name(physics_cfg: Any) -> str | None:
    """Name the concrete physics config a run resolved to, solver included.

    Newton's solver lives on ``solver_cfg`` and is what separates ``newton_mjwarp``
    from ``newton_kamino``; the class name alone reports both as ``NewtonCfg``.
    """
    if physics_cfg is None:
        return None
    solver = getattr(physics_cfg, "solver_cfg", None)
    name = type(physics_cfg).__name__
    return f"{name}({type(solver).__name__})" if solver is not None else name


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

    A task declaring renderers is expanded across them: reporting a camera task as
    headless-only omits the thing under test. Domain presets are expanded one at a
    time and never combined, so every preset name is validated on its own but no pair
    is ever tried (see :class:`DiscoveredTask.Mode`).

    With ``collapse``, the cross product is deduplicated on the resolved config, so
    each returned mode is a distinct run rather than a distinct spelling. That is what
    removes selector double-counting without a table of alias names: ``physics=physx``
    and ``physics=ovphysx`` collapse wherever they resolve alike and stay separate
    wherever they do not. Collapsing on the *backend* instead would be wrong — the
    Reach controller presets share a backend and are four different runs.

    Without ``collapse``, every combination that validated is returned, duplicate
    spellings included. That is what documentation wants: ``presets=shapes`` is a real
    token a reader can type even on a task where it happens to name the default of its
    own axis, and collapsing would delete it from the table.

    Declared mode neither validates nor collapses, because nothing has been resolved:
    it returns the raw cross product and no default.

    Returns:
        ``(modes, default)``. *default* is the run the task performs when the user
        passes nothing, or ``None`` in declared mode / when that run cannot resolve.
        It names an explicit spelling even when ``collapse`` is off.

    Raises:
        DiscoveryError: If ``strict`` and the validator could not judge a combination.
    """
    physics_options: tuple[str | None, ...] = physics or (None,)
    renderer_options: tuple[str | None, ...] = renderers or (None,)
    # ``None`` is the task's own default. It survives the collapse only when it is a
    # run of its own; usually it folds into whichever preset the config defaults to.
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
    # The no-token run is what a user gets by changing nothing, so it always has to be
    # probed. It is not always in the cross product: a task declaring physics presets
    # has no ``physics=None`` column, only its declared backends.
    if (None, None, None) not in combinations:
        combinations.insert(0, (None, None, None))

    # ``unique`` is built either way: even when every validated mode is returned, it is
    # what identifies the explicit spelling of the no-token run.
    unique: dict[str, DiscoveredTask.Mode] = {}
    validated: list[DiscoveredTask.Mode] = []
    default_key: str | None = None
    default_backend: str | None = None
    for physics_name, renderer, domain in combinations:
        try:
            resolution = _mode_resolves(task_id, physics_name, renderer, domain)
        except DiscoveryError as exc:
            if strict:
                raise
            # Unknown is not the same answer as legal, so the combination is dropped
            # -- but loudly, because it is a gap in the matrix.
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
        # Keep the most explicit spelling of each run: naming the presets reproduces
        # it even if the config's own defaults move later.
        incumbent = unique.get(key)
        if incumbent is None or _explicitness(mode) > _explicitness(incumbent):
            unique[key] = mode

    default = None
    if default_key is not None:
        default = DiscoveredTask.Default(backend=default_backend, mode=unique[default_key])
    return tuple(unique.values()) if collapse else tuple(validated), default


def _explicitness(mode: DiscoveredTask.Mode) -> int:
    """Return how many selector tokens *mode* spells out."""
    return sum(token is not None for token in (mode.physics, mode.renderer, mode.presets))


def discover_tasks(
    specs: list[Any] | None = None, *, resolve: bool = True, strict: bool = False, collapse: bool = True
) -> list[DiscoveredTask]:
    """Walk the Gym registry and return every registered training task.

    Imports Isaac Lab, so it needs the project environment. Contrib tasks are included
    when ``isaaclab_tasks_experimental`` is importable.

    Args:
        specs: Gym specs to walk. When ``None``, the whole registry is scanned.
        resolve: When ``True``, every backend combination is built and checked against
            the runtime validator, and only combinations that can actually run are
            returned. When ``False``, combinations are reported as declared, which is
            fast but unverified.
        strict: When ``True``, a task the validator cannot run against at all raises
            instead of being logged and skipped. Off by default so that one broken
            task costs the caller that task rather than the whole registry; turn it
            on to police Isaac Lab API drift.
        collapse: When ``True`` (the default), spellings that resolve to the same
            config are reduced to one, so ``modes`` is the list of distinct runs a
            dispatcher should schedule. Turn it off to keep every validated spelling —
            what documentation needs, since a preset that names the default of its own
            axis is still a token a reader can type. Ignored when ``resolve`` is off.

    Returns:
        Discovered tasks sorted by ``task_id``.

    Raises:
        DiscoveryError: If the task packages cannot be imported, or, when ``strict``,
            if any task could not be inspected.
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
        # Tasks without an RL entry point (IK, teleop, mimic) are still registered
        # environments and are reported with an empty ``rl_libraries``. Callers that
        # need a trainable task — a dispatcher, say — filter on it themselves rather
        # than having that policy baked in here.
        libraries = _rl_libraries_from_kwargs(spec.kwargs)
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
                rl_libraries=libraries,
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
