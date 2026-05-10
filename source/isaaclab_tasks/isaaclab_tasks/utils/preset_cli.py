# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed CLI flags for preset selection.

Pure translator: per-:class:`PresetTarget` argparse flag → ``presets=<csv>``
token in ``sys.argv``. The Hydra-decorator flow in
:mod:`isaaclab_tasks.utils.hydra` is unchanged and consumes the token via
its existing ``register_task`` / ``apply_overrides`` path.

Most scripts use the one-line form::

    parser = argparse.ArgumentParser(...)
    # ... script-specific args ...
    args_cli = setup_cli(parser)

How the registry gets populated: backends register themselves via the
``@register(target, name)`` decorator on their cfg classes. The decorator
fires only when that cfg module is imported. Plain ``import isaaclab_tasks``
does *not* transitively import backend cfgs; they're referenced inside env
config classes via ``from isaaclab_physx.physics import PhysxCfg``-style
imports. ``setup_cli`` therefore looks up ``--task=X`` in argv and loads
that task's env config to trigger the chain. After that, the registry is
populated for whichever backends X uses, and validation + ``--help``
listings are accurate.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.utils.preset_registry import PresetRegistry, PresetTarget


def _extract_task_from_argv(argv: list[str]) -> str | None:
    """Best-effort scan of *argv* for ``--task=value`` / ``--task value``.

    Used before argparse parses, so we can pre-load the task's env config
    (which transitively imports its backends and populates the registry)
    in time for ``--help`` enrichment and validation.
    """
    for i, token in enumerate(argv):
        if token == "--task" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--task="):
            return token[len("--task=") :]
    return None


def _load_task_backends(task_name: str) -> None:
    """Load *task_name*'s env config so its backends register themselves.

    Silent on failure: validation / help will report empty registries
    rather than raise here. The actual error (e.g., unknown task) will
    surface from Hydra later with a more informative message.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    try:
        load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")
    except Exception:
        # Don't break --help / validation just because the task name
        # is bad or its config has an import-time error; let Hydra
        # surface that failure when it tries to register the task.
        pass


def _validate_typed_flag(target: PresetTarget, value: str | None) -> str | None:
    """Reject unknown canonical names; normalize legacy aliases.

    Caller must have already loaded the task's backends (via
    :func:`_load_task_backends`) so the registry reflects what's
    available for the current task. Returns the canonical name (possibly
    normalized from a legacy alias) or ``None`` when *value* is ``None``.
    Raises ``SystemExit`` with a helpful message when the name is not
    registered for *target*.
    """
    if value is None:
        return None
    canonical = target.normalize(value)
    valid = PresetRegistry.names_for(target)
    if canonical not in valid:
        listing = ", ".join(sorted(valid)) if valid else "(no backends registered for this task)"
        raise SystemExit(
            f"error: --{target.value} {value!r} is not a recognized {target.value} preset.\n"
            f"  Valid {target.value} presets: {listing}"
        )
    return canonical


class _HelpAction(argparse._HelpAction):
    """Help action that lists registered preset names after standard help.

    Triggered by ``-h`` / ``--help``. Before printing, scans argv for
    ``--task=X`` and loads X's env config so the registry reflects what
    that task's backends provide.
    """

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        task = _extract_task_from_argv(sys.argv[1:])
        if task:
            _load_task_backends(task)
        parser.print_help()
        sys.stdout.write("\navailable preset names")
        if task:
            sys.stdout.write(f" (for task {task!r})")
        sys.stdout.write(":\n")
        for target in PresetTarget:
            if target is PresetTarget.DOMAIN:
                # Domain is free-form; no canonical vocabulary to list.
                continue
            names = sorted(PresetRegistry.names_for(target))
            listing = ", ".join(names) if names else (
                "(no backends loaded; pass --task=<X> to populate)" if not task else "(none for this task)"
            )
            sys.stdout.write(f"  --{target.value}: {listing}\n")
        parser.exit()


def setup_cli(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Add per-target preset flags + AppLauncher flags, parse, fold into ``presets=<csv>``.

    Steps:

    1. For every :class:`PresetTarget`: register one argparse flag.
       Non-DOMAIN targets get ``--{target.value}=NAME`` (single canonical
       name). DOMAIN gets ``--presets=NAME[,NAME,...]`` (free-form CSV).
    2. Replace argparse's default ``_HelpAction`` with one that
       lists registered preset names after the standard help, scoped to
       ``--task=X`` if the user supplied one.
    3. Register AppLauncher flags via ``AppLauncher.add_app_launcher_args``.
    4. Call ``parser.parse_known_args``; argparse-handled tokens
       disappear, Hydra-style ``key=value`` tokens stay in *remaining*.
    5. If any typed preset flag is set: load the task's env config so its
       backends register, then validate each flag against the registry.
    6. Collect names + merge with any pre-existing ``presets=...`` token
       from *remaining*; dedupe; rewrite ``sys.argv`` so the Hydra layer
       sees one ``presets=<csv>`` token followed by leftover Hydra args.

    Returns the parsed argparse namespace; ``sys.argv`` is mutated in place.
    """
    # Lazy: AppLauncher pulls in Isaac Sim. Keep the import inside this
    # function so this module is importable without Sim being available.
    from isaaclab.app import AppLauncher

    group = parser.add_argument_group(
        "preset selection",
        description=(
            "Select named PresetCfg alternatives at runtime. Both '--flag value'"
            " and '--flag=value' are accepted. Flags translate to a 'presets=<csv>'"
            " token consumed by the Hydra-decorator flow. Run with '--task=<X> --help'"
            " to list valid names for a specific task."
        ),
    )
    for target in PresetTarget:
        if target is PresetTarget.DOMAIN:
            group.add_argument(
                "--presets",
                type=str,
                default=None,
                metavar="NAME[,NAME,...]",
                help="Comma-separated free-form preset names (broadcast to every matching PresetCfg).",
            )
        else:
            group.add_argument(
                f"--{target.value}",
                type=str,
                default=None,
                metavar="NAME",
                help=f"{target.value.capitalize()} preset name (use '--task=<X> --help' to list valid names).",
            )

    # Swap argparse's default --help action with one that enriches output
    # by loading the task's env config + listing registered preset names.
    for idx, action in enumerate(parser._actions):
        if isinstance(action, argparse._HelpAction) and not isinstance(action, _HelpAction):
            replacement = _HelpAction(
                option_strings=list(action.option_strings),
                dest=action.dest,
                default=action.default,
                help=action.help,
            )
            parser._actions[idx] = replacement
            for opt in action.option_strings:
                parser._option_string_actions[opt] = replacement
            break

    AppLauncher.add_app_launcher_args(parser)
    args, remaining = parser.parse_known_args()

    # Detect whether ANY typed flag is set; if so we'll need to validate
    # against the registry, which means the task's backends must be loaded.
    typed_values = {target: getattr(args, target.value) for target in PresetTarget if target is not PresetTarget.DOMAIN}
    any_typed = any(value is not None for value in typed_values.values())

    if any_typed:
        if not args.task:
            raise SystemExit(
                "error: --physics/--renderer require --task=<task-name> to validate against."
            )
        _load_task_backends(args.task)

    # Collect everything the user asked for, in declaration order.
    names: list[str] = []
    for target in PresetTarget:
        if target is PresetTarget.DOMAIN:
            # Free-form CSV; trust whatever the user typed.
            raw = args.presets
            if raw:
                names.extend(name.strip() for name in raw.split(",") if name.strip())
        else:
            canonical = _validate_typed_flag(target, typed_values[target])
            if canonical:
                names.append(canonical)

    if not names:
        # Nothing preset-related; pass argv through unchanged.
        sys.argv = [sys.argv[0], *remaining]
        return args

    # Merge with any pre-existing ``presets=...`` token in remaining.
    kept: list[str] = []
    for token in remaining:
        if token.startswith("presets="):
            names.extend(name.strip() for name in token[len("presets=") :].split(",") if name.strip())
        else:
            kept.append(token)

    # Dedupe, preserve first-occurrence order.
    seen: set[str] = set()
    deduped = [name for name in names if not (name in seen or seen.add(name))]
    sys.argv = [sys.argv[0], f"presets={','.join(deduped)}", *kept]
    return args
