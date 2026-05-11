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

Two sources populate the set of accepted preset names:

* The decorator registry, populated when backend cfg modules are imported.
  ``@register(target, name)`` is the canonical declaration.
* The currently selected task's :class:`PresetCfg` field names. A user can
  add a *variant* alternative without re-decorating::

      @configclass
      class PhysicsCfg(PresetCfg):
          default: ... = MISSING
          newton_mjwarp: MjwarpCfg = MjwarpCfg()  # canonical
          newton_mjwarp_strict: MjwarpCfg = MjwarpCfg(...)  # variant

  ``--physics newton_mjwarp_strict`` is accepted and selects the variant.

How the registry gets populated: backends register themselves via the
``@register(target, name)`` decorator on their cfg classes. The decorator
fires only when that cfg module is imported. Plain ``import isaaclab_tasks``
does *not* transitively import backend cfgs; they're referenced inside env
config classes via ``from isaaclab_physx.physics import PhysxCfg``-style
imports. ``setup_cli`` therefore looks up ``--task=X`` in argv and loads
that task's env config to trigger the chain. After that, the registry is
populated for whichever backends X uses, ``--help`` listings include both
canonical names and task variants, and validation messages can list valid
choices.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.utils.preset_registry import PresetRegistry, PresetTarget

# ============================================================================
# Pre-parse setup: scan argv for --task, load that task's env config
# ============================================================================


def _extract_task_from_argv(argv: list[str]) -> str | None:
    """Best-effort scan of *argv* for ``--task=value`` / ``--task value``.

    Used before argparse parses, so we can pre-load the task's env config
    (which transitively imports its backends and populates the registry)
    in time for ``--help`` enrichment and validation. The returned value
    is the single source of truth for the rest of ``setup_cli``; a guard
    after argparse runs rejects mismatched argparse values.

    Stops at the ``--`` end-of-options marker, matching argparse's
    semantics for positional separation. Picks the last occurrence on
    repetition to match argparse's last-wins behavior.

    Limitation: argparse's default ``allow_abbrev=True`` accepts
    unambiguous prefixes (e.g. ``--tas Foo``); the pre-scan only
    recognizes the literal ``--task`` / ``--task=``. :func:`setup_cli`
    detects the mismatch after argparse runs and raises ``SystemExit``
    asking for the full spelling (or pass ``allow_abbrev=False`` to
    your parser).

    Args:
        argv: Token list to scan, typically ``sys.argv[1:]``.

    Returns:
        The value paired with the rightmost ``--task`` token before any
        ``--`` separator, or ``None`` when no ``--task`` appears.
    """
    last: str | None = None
    for i, token in enumerate(argv):
        if token == "--":
            break
        # ``--task --`` is a syntax error to argparse (no value); don't
        # consume ``--`` as the value here.
        if token == "--task" and i + 1 < len(argv) and argv[i + 1] != "--":
            last = argv[i + 1]
        elif token.startswith("--task="):
            last = token[len("--task=") :]
    return last


def _load_task_env_cfg(task_name: str) -> object | None:
    """Look up and instantiate the env config registered for *task_name*.

    Side effect: importing the env config module triggers the backend cfg
    imports referenced by the task, which fires the :func:`register`
    decorators that populate :class:`PresetRegistry`.

    Scope: this feature only handles its own input. Failures that
    genuinely belong to gym / Isaac Sim / the task author (a misspelled
    task name, a buggy task-config module, a config ``__init__`` that
    raises) propagate as-is so the user sees the real error rather than
    a misleading "not a recognized preset" message later.

    The single tolerated failure is missing Isaac Sim runtime deps
    (``ImportError`` / ``ModuleNotFoundError``) in headless or CI
    environments. There we emit a stderr note and degrade to "validate
    only against names already registered" so ``--help`` and built-in
    canonical names remain usable without the full Sim stack.

    Args:
        task_name: Gym-style task id, optionally with a ``"namespace:"``
            prefix that is stripped before registry lookup.

    Returns:
        The instantiated env config, or ``None`` when the only tolerated
        failure (a missing Isaac Sim runtime dep) tripped the load.

    Raises:
        Exception: Any non-import failure from the task lookup or the
            config class's ``__init__`` propagates verbatim.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    try:
        env_cfg = load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")
    except (ImportError, ModuleNotFoundError) as exc:
        sys.stderr.write(
            f"warning: backend deps unavailable while loading task {task_name!r} for preset validation: "
            f"{type(exc).__name__}: {exc}. Falling back to registered names only.\n"
        )
        return None
    if isinstance(env_cfg, type):
        env_cfg = env_cfg()  # let __init__ failures propagate -- not our problem
    return env_cfg


# ============================================================================
# Per-flag handling: help string (used at arg-registration time) +
# validation (used at name-collection time after argparse has run)
# ============================================================================


def _help_text(target: PresetTarget, valid: set[str], task: str | None) -> str:
    """Build the argparse ``help=`` string for *target*'s flag.

    Args:
        target: Which preset target's flag is being described
            (determines the capitalized prefix).
        valid: Names to enumerate in the listing -- typically the union
            of registered canonical names and task variants for
            *target*. An empty set produces a hint to pass ``--task``.
        task: When supplied, the listing is scoped ("for task X")
            instead of described as "registered". This is the
            pre-scanned task id, not the post-argparse one, since the
            help fires before argparse runs.

    Returns:
        A single-line string suitable to pass as
        ``add_argument(..., help=...)``.
    """
    capitalized = target.value.capitalize()
    if not valid:
        return f"{capitalized} preset name. Pass '--task=<X> --help' to list valid names for task X."
    listing = ", ".join(sorted(valid))
    scope = f"for task {task!r}" if task else "registered"
    return f"{capitalized} preset name. Available ({scope}): {listing}."


def _validate_typed_flag(target: PresetTarget, value: str | None, variants: set[str]) -> str | None:
    """Reject unknown names; normalize legacy aliases.

    A name is valid when it is in :meth:`PresetRegistry.names_for` for
    *target* (a registered backend) or in *variants* (a field name in the
    selected task's :class:`PresetCfg` for *target*). A task-local variant
    that happens to share a deprecated alias name (e.g. a real ``newton``
    field on the task's ``PhysicsCfg``) is preserved as-is and *not*
    rewritten to the alias's canonical -- the variant shadows the alias.

    Args:
        target: Which preset target the value belongs to (drives the
            registry lookup and alias map).
        value: Whatever the user typed after ``--{target.value}`` -- or
            ``None`` when they didn't pass the flag.
        variants: Names accepted as variants for *target* on the
            currently loaded task. Pass an empty set when no task is
            loaded (validation then collapses to "registered names
            only"). Typically ``task_variants.get(target, set())``.

    Returns:
        The name to emit in the ``presets=<csv>`` token: *value*
        unchanged when it's a task variant, the canonical replacement
        when *value* was a legacy alias, or ``value`` itself when it
        was already canonical. ``None`` when *value* is ``None``.

    Raises:
        SystemExit: *value* is neither a registered canonical for
            *target*, a task variant, nor a legacy alias normalizing
            into either; the message lists the names that *would* have
            been accepted.
    """
    if value is None:
        return None
    # Variant shadows alias: a real field named 'newton' on this task's
    # PresetCfg means the user wants that field, not the deprecated alias.
    if value in variants:
        return value
    canonical = target.normalize(value)
    valid = PresetRegistry.names_for(target) | variants
    if canonical not in valid:
        listing = ", ".join(sorted(valid)) if valid else "(no presets registered for this task)"
        raise SystemExit(
            f"error: --{target.value} {value!r} is not a recognized {target.value} preset.\n"
            f"  Valid {target.value} presets: {listing}"
        )
    return canonical


# ============================================================================
# Entry point
# ============================================================================


def setup_cli(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Add per-target preset flags + AppLauncher flags, parse, fold into ``presets=<csv>``.

    Steps:

    1. Pre-scan ``sys.argv`` for ``--task=X``. If found, load the task's
       env config so its backends register themselves (populating
       :class:`PresetRegistry`) and harvest variant field names from its
       :class:`PresetCfg` instances.
    2. For every :class:`PresetTarget`: register one argparse flag.
       Non-DOMAIN targets get ``--{target.value}=NAME`` whose ``help=``
       lists the valid names for the current task. DOMAIN gets
       ``--presets=NAME[,NAME,...]`` (free-form CSV).
    3. Register AppLauncher flags via ``AppLauncher.add_app_launcher_args``.
    4. Call ``parser.parse_known_args``; argparse-handled tokens
       disappear, Hydra-style ``key=value`` tokens stay in *remaining*.
    5. Sanity-guard that the parsed ``--task`` matches the pre-scan value
       (they only disagree when argparse expanded an abbreviation the
       pre-scan didn't recognize); raise if not.
    6. Validate each typed flag against ``names_for(target) | variants[target]``.
    7. Collect names + merge with any pre-existing ``presets=...`` token
       from *remaining*; dedupe; rewrite ``sys.argv`` so the Hydra layer
       sees one ``presets=<csv>`` token followed by leftover Hydra args.

    Args:
        parser: The caller's argument parser. Preset flags and the
            AppLauncher flag set are added to it in place; callers do
            NOT need to register either themselves.

    Returns:
        The namespace from ``parse_known_args``. The function also
        mutates ``sys.argv`` in place so the Hydra layer downstream
        sees the folded ``presets=<csv>`` token followed by whatever
        the parser didn't consume.

    Raises:
        SystemExit: A typed flag was set without ``--task``; the parsed
            ``--task`` disagrees with the pre-scan (argparse
            abbreviation); or a typed flag's value isn't a recognized
            name for its target.
    """
    # Lazy: AppLauncher pulls in Isaac Sim. Keep the import inside this
    # function so this module is importable without Sim being available.
    from isaaclab.app import AppLauncher

    # Pre-scan: load the task's env config now so the registry + variant
    # set are ready when we build help strings and validate. ``pre_task``
    # is the single source of truth for the rest of this function; the
    # guard below catches the only realistic disagreement (argparse
    # abbreviations like ``--tas Foo`` that the pre-scan deliberately
    # doesn't recognize).
    from isaaclab_tasks.utils.hydra import collect_task_variants as _collect_task_variants

    pre_task = _extract_task_from_argv(sys.argv[1:])
    env_cfg = _load_task_env_cfg(pre_task) if pre_task else None
    task_variants: dict[PresetTarget, set[str]] = _collect_task_variants(env_cfg) if env_cfg is not None else {}

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
            valid = PresetRegistry.names_for(target) | task_variants.get(target, set())
            group.add_argument(
                f"--{target.value}",
                type=str,
                default=None,
                metavar="NAME",
                help=_help_text(target, valid, pre_task),
            )

    AppLauncher.add_app_launcher_args(parser)
    args, remaining = parser.parse_known_args()

    # Detect whether ANY typed flag is set; if so we'll validate against
    # registry ∪ task variants, both of which assume the task is loaded.
    typed_values = {target: getattr(args, target.value) for target in PresetTarget if target is not PresetTarget.DOMAIN}
    any_typed = any(value is not None for value in typed_values.values())

    # Sanity guard: when argparse parsed ``--task``, its value must match
    # the pre-scan. They only disagree when the user used an argparse
    # abbreviation (e.g. ``--tas Foo``) that the pre-scan deliberately
    # doesn't recognize -- raise instead of silently re-loading so a
    # single source of truth exists for the rest of this function.
    parsed_task = getattr(args, "task", None)
    if parsed_task is not None and parsed_task != pre_task:
        raise SystemExit(
            f"error: --task value {parsed_task!r} doesn't match the pre-scan value {pre_task!r}. "
            "preset-CLI requires the literal '--task=NAME' form (argparse-style abbreviations like "
            "'--tas' are not supported)."
        )

    if any_typed and not pre_task:
        typed_flags = ", ".join(f"--{t.value}" for t in PresetTarget if t is not PresetTarget.DOMAIN)
        raise SystemExit(f"error: typed preset flags ({typed_flags}) require --task=<task-name> to validate against.")

    # Collect everything the user asked for, in declaration order.
    names: list[str] = []
    for target in PresetTarget:
        if target is PresetTarget.DOMAIN:
            # Free-form CSV; trust whatever the user typed.
            raw = args.presets
            if raw:
                names.extend(name.strip() for name in raw.split(",") if name.strip())
        else:
            canonical = _validate_typed_flag(target, typed_values[target], task_variants.get(target, set()))
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
