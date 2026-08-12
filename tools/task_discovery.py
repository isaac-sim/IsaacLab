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

The gap between the two is itself useful: a combination that is declared but does not
resolve is documentation drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RL_LIBRARY_PRIORITY",
    "DiscoveredTask",
    "DiscoveryError",
    "discover_tasks",
]

# Stable ordering for the RL library axis.
RL_LIBRARY_PRIORITY: tuple[str, ...] = ("rsl_rl", "rl_games", "skrl", "sb3")

# Physics presets that are proxy variants rather than backends under test.
_SKIP_PHYSICS = frozenset({"newton_mjwarp_vbd_proxy"})

# Backend names that also appear under ``PresetTarget.DOMAIN`` on some tasks.
# They are selected with ``physics=`` / ``renderer=``, so reporting them as a
# ``presets=`` token would be a duplicate at best and wrong at worst. A per-task
# check is not enough: a task can list ``newton_mjwarp`` under DOMAIN without
# declaring it under PHYSICS.
_BACKEND_MIRROR_NAMES = frozenset(
    {
        "newton_kamino",
        "newton_mjwarp",
        "newton_mjwarp_vbd",
        "newton_mjwarp_vbd_proxy",
        "ovphysx",
        "physx",
        "isaacsim_physx",
        "newton",
        "kamino",
        "isaacsim_rtx",
        "isaacsim_rtx_renderer",
        "newton_renderer",
        "ovrtx",
        "ovrtx_renderer",
        "rtx",
    }
)

# Errors that mean the validator itself could not run, rather than that the
# combination under test was rejected. Swallowing these marks every combination
# illegal and leaves callers blaming their filters for an empty result.
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
            ``renderer``, ``presets``). Backend mirrors are already removed from
            ``presets``.
        selectors: Declared names that are automatic selectors rather than concrete
            backends, keyed by axis. ``physics=physx`` and ``renderer=rtx`` resolve to
            a backend at launch, so a selector and its target are the same run.
            Consumers that must not double-count — a benchmark dispatcher — subtract
            these; the environment tables exclude them for the same reason.
        modes: Backend combinations. In resolved mode these are the combinations that
            passed the runtime validator; in declared mode they are the full cross
            product, unverified.
        resolved: Whether ``modes`` was filtered by the runtime validator.
    """

    @dataclass(frozen=True)
    class Mode:
        """One way to run a task.

        Args:
            physics: Physics preset token, or ``None`` for tasks that declare none
                and reject any ``physics=`` selector.
            renderer: Renderer preset token, or ``None`` to run headless.
            presets: Domain preset token passed as ``presets=<value>``, or ``None``.
                Exactly one at a time: domain presets targeting the same field
                conflict outright, e.g. ``presets=depth,rgb`` is rejected.
        """

        physics: str | None
        renderer: str | None
        presets: str | None

    task_id: str
    scope: str
    rl_libraries: tuple[str, ...]
    declared: dict[str, tuple[str, ...]] = field(default_factory=dict)
    selectors: dict[str, tuple[str, ...]] = field(default_factory=dict)
    modes: tuple[DiscoveredTask.Mode, ...] = ()
    resolved: bool = False


def _selector_names(task_id: str) -> dict[str, tuple[str, ...]]:
    """Return the declared preset names that are automatic selectors, by axis.

    A selector is an alias rather than a backend: ``physics=physx`` resolves to
    OvPhysX kitless and to Isaac Sim PhysX under Kit, and ``renderer=rtx`` picks an
    RTX backend the same way. Because the target depends on how the run is
    launched, a selector and the backend it resolves to are the same run — which is
    why the environment tables list only concrete backends.

    Detected by config type rather than by name so that adding a selector upstream
    does not require editing a hardcoded list here.
    """
    from isaaclab.physics.physics_manager_cfg import PhysxAutoCfg

    from isaaclab_tasks.utils.hydra import collect_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    try:
        walked = collect_presets(load_cfg_from_registry(task_id, "env_cfg_entry_point"))
    except _INFRASTRUCTURE_ERRORS as exc:
        # A structural failure here returns no selectors, which is indistinguishable
        # from a task that genuinely has none — so it would silently disable
        # selector-aware filtering for every task. Fail loudly instead.
        raise DiscoveryError(f"selector detection for {task_id!r} could not run: {type(exc).__name__}: {exc}") from exc
    except Exception:  # noqa: BLE001 - a config that cannot load declares no selectors
        return {}

    # ``collect_presets`` maps dotted config paths to ``{preset name: cfg}``, so the
    # variants live one level in. Iterating the outer mapping yields dicts, never
    # configs, and silently finds nothing.
    physics: set[str] = set()
    renderer: set[str] = set()
    for variants in walked.values():
        for name, cfg in variants.items():
            if isinstance(cfg, PhysxAutoCfg):
                physics.add(name)
            elif getattr(cfg, "renderer_type", None) == "auto_rtx":
                renderer.add(name)
    return {"physics": tuple(sorted(physics)), "renderer": tuple(sorted(renderer))}


def _canonical_physics(names: tuple[str, ...]) -> tuple[str, ...]:
    """Drop proxy physics variants that are never a backend under test.

    ``physx`` is deliberately kept even when ``ovphysx`` is also declared. It is a
    real selector that resolves to a concrete backend, so a task matrix should
    report it. Callers that would run both and consider the pair redundant — a
    benchmark dispatcher, say — should apply that policy themselves.
    """
    return tuple(name for name in names if name not in _SKIP_PHYSICS)


def _domain_presets(names: list[str]) -> tuple[str, ...]:
    """Return domain presets, dropping names that mirror a backend selector."""
    return tuple(sorted(name for name in names if name not in _BACKEND_MIRROR_NAMES))


def _is_training_task(task_id: str) -> bool:
    """Return whether *task_id* is a trainable Isaac task."""
    if "Isaac" not in task_id:
        return False
    return not task_id.endswith("-Eval") and "-Benchmark-" not in task_id


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


def _mode_resolves(task_id: str, physics: str | None, renderer: str | None, presets: str | None = None) -> bool:
    """Return whether a physics/renderer/preset combination resolves and validates.

    An unknown preset, an unloadable config, or a rejected backend combination all
    mean the same thing — the combination cannot run — so they return ``False`` alike.

    Raises:
        DiscoveryError: If validation could not run at all, e.g. because an Isaac Lab
            import or API it depends on has changed.
    """
    import argparse
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
        return True
    except _INFRASTRUCTURE_ERRORS as exc:
        raise DiscoveryError(
            f"preset validation for {task_id!r} could not run: {type(exc).__name__}: {exc}. This is an Isaac Lab"
            " import or API failure, not a rejected preset combination."
        ) from exc
    except Exception:  # noqa: BLE001 - any other failure means the combination cannot run
        return False
    finally:
        sys.argv = original_argv


def _build_modes(
    task_id: str,
    physics: tuple[str, ...],
    renderers: tuple[str, ...],
    domains: tuple[str, ...],
    *,
    resolve: bool,
) -> tuple[DiscoveredTask.Mode, ...]:
    """Return the backend combinations for one task.

    A task declaring renderers is expanded across them: reporting a camera task as
    headless-only omits the thing under test. Domain presets are expanded one at a
    time, never combined, because presets targeting the same field conflict.
    """
    physics_options: tuple[str | None, ...] = physics or (None,)
    renderer_options: tuple[str | None, ...] = renderers or (None,)
    # ``None`` keeps the task's own default alongside each explicit preset.
    domain_options: tuple[str | None, ...] = (None, *domains) if domains else (None,)

    modes: list[DiscoveredTask.Mode] = []
    for physics_name in physics_options:
        for renderer in renderer_options:
            for domain in domain_options:
                if resolve and not _mode_resolves(task_id, physics_name, renderer, domain):
                    continue
                modes.append(DiscoveredTask.Mode(physics=physics_name, renderer=renderer, presets=domain))
    return tuple(modes)


def discover_tasks(*, resolve: bool = True) -> list[DiscoveredTask]:
    """Walk the Gym registry and return every registered training task.

    Imports Isaac Lab, so it needs the project environment. Contrib tasks are included
    when ``isaaclab_tasks_experimental`` is importable.

    Args:
        resolve: When ``True``, every backend combination is built and checked against
            the runtime validator, and only combinations that can actually run are
            returned. When ``False``, combinations are reported as declared, which is
            fast but unverified.

    Returns:
        Discovered tasks sorted by ``task_id``.

    Raises:
        DiscoveryError: If the task packages cannot be imported.
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

    tasks: list[DiscoveredTask] = []
    for spec in gym.registry.values():
        if not _is_training_task(spec.id) or spec.kwargs.get("deprecated"):
            continue
        # Tasks without an RL entry point (IK, teleop, mimic) are still registered
        # environments and are reported with an empty ``rl_libraries``. Callers that
        # need a trainable task — a dispatcher, say — filter on it themselves rather
        # than having that policy baked in here.
        libraries = _rl_libraries_from_kwargs(spec.kwargs)
        preset_map = enumerate_task_presets(spec.id)
        physics = _canonical_physics(tuple(sorted(preset_map.get(PresetTarget.PHYSICS, [])))) if preset_map else ()
        renderers = tuple(sorted(preset_map.get(PresetTarget.RENDERER, []))) if preset_map else ()
        domains = _domain_presets(preset_map.get(PresetTarget.DOMAIN, [])) if preset_map else ()
        tasks.append(
            DiscoveredTask(
                task_id=spec.id,
                scope="contrib" if spec.id.startswith("IsaacContrib-") else "core",
                rl_libraries=libraries,
                declared={"physics": physics, "renderer": renderers, "presets": domains},
                selectors=_selector_names(spec.id),
                modes=_build_modes(spec.id, physics, renderers, domains, resolve=resolve),
                resolved=resolve,
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    return tasks
