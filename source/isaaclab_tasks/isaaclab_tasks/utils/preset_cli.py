# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed CLI flags for preset selection.

Adds ``--physics`` / ``--renderer`` / ``--presets`` argparse flags to
the user's parser and folds their values into the same ``presets=<csv>``
token that the existing Hydra-decorator preset flow already consumes
via :mod:`isaaclab_tasks.utils.hydra`. This module is a pure translator:

* It does **not** validate names at CLI time. A user-typed name that
  isn't in the registry might still be a legitimate task-local preset
  (e.g. ``--presets=cube`` for Dexsuite's ``ObjectCfg.cube``). Only the
  resolver has the loaded task's full vocabulary and can tell typos
  apart from custom names; it produces the rich path-grouped error via
  :func:`isaaclab_tasks.utils.hydra._format_unknown_presets_error` at
  resolve time.
* It does **not** rewrite legacy aliases. Hydra's
  :func:`isaaclab_tasks.utils.hydra._normalize_preset_name` rewrites
  them at resolve time with a ``FutureWarning``.
* It does **not** resolve presets. The existing
  :func:`isaaclab_tasks.utils.hydra.resolve_presets` does that.

The single new capability is **discoverability**: when ``--task=X`` is
present in ``sys.argv``, ``--help`` loads ``X``'s env_cfg, walks its
:class:`PresetCfg` fields via
:func:`isaaclab_tasks.utils.hydra.collect_presets`, and shows the
variants actually present in that task -- not a static registry. The
load is safe before ``AppLauncher`` boots because IsaacLab's
``test_env_cfg_no_forbidden_imports.py`` enforces that env_cfg modules
do not import ``pxr`` / ``omni`` / ``carb`` / ``isaacsim`` at top
level. Without ``--task``, ``--help`` tells the user to pass one
(the available variants are task-dependent and we don't try to guess).

Typical script setup::

    parser = argparse.ArgumentParser(...)
    # ... script-specific args ...
    add_launcher_args(parser)  # AppLauncher flags (--headless, --device, ...)
    args_cli = setup_preset_cli(parser)  # preset flags + parse + fold sys.argv

``setup_preset_cli`` does NOT add AppLauncher flags itself -- callers add them
explicitly via :func:`isaaclab_tasks.utils.add_launcher_args` before
calling ``setup_preset_cli``. Two reasons: ``setup_preset_cli`` then has a single
responsibility (preset CLI), and scripts that already call
``add_launcher_args`` can adopt ``setup_preset_cli`` without first removing
their existing call (which would otherwise collide with a duplicate
``--headless`` registration).
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.utils.preset_registry import PresetRegistry, PresetTarget


def setup_preset_cli(parser: argparse.ArgumentParser) -> tuple[argparse.Namespace, list[str]]:
    """Add typed preset flags, parse, and compute the Hydra-bound argv tokens.

    Steps:

    1. If ``sys.argv`` contains ``--task=X`` and ``--help`` was also
       requested, load ``X``'s env_cfg and enumerate its
       :class:`PresetCfg` variants for use in help text (see
       :func:`_enumerate_variants`).
    2. Register one argparse flag per :class:`PresetTarget`:
       ``--{target.value}=NAME`` for typed targets, ``--presets=NAME[,NAME,...]``
       for the DOMAIN catch-all. Help strings list variants from step 1,
       or tell the user to pass ``--task=X`` if no task was given.
    3. Call ``parser.parse_known_args``.
    4. Pass typed and free-form values through verbatim. Fold them
       (plus any pre-existing ``presets=...`` token in *remaining*) into
       a single ``presets=<csv>`` token. Return the resulting token list so
       the caller can assign it to ``sys.argv[1:]`` when it's ready.

    Callers must add AppLauncher flags (via
    :func:`isaaclab_tasks.utils.add_launcher_args`) and any
    script-specific arguments *before* calling this function -- otherwise
    those unknown tokens land in ``parse_known_args``'s remainder.

    This function deliberately does NOT mutate ``sys.argv``. Mutation is
    the caller's responsibility (typical pattern: ``sys.argv = [sys.argv[0]]
    + hydra_argv``). The deferral lets scripts insert argv-aware logic
    (e.g., an ``--external_callback`` hook that re-reads ``sys.argv``)
    between parse and the final ``sys.argv`` assignment, where folding
    earlier would hide the user's original command line from the callback.

    Args:
        parser: The caller's argument parser. Preset flags are added in
            place. Must already have AppLauncher flags and any
            script-specific arguments registered.

    Returns:
        ``(args, hydra_argv)`` where ``args`` is the namespace from
        ``parse_known_args`` and ``hydra_argv`` is the list of tokens to
        hand to Hydra via ``sys.argv[1:]``. ``hydra_argv[0]`` is a folded
        ``presets=<csv>`` token whenever any preset flag (typed or
        free-form) was given or a pre-existing ``presets=...`` token was
        present in the remainder; otherwise the list contains only the
        non-preset remainder.
    """
    # Peek for --task before argparse parses. argparse short-circuits on --help,
    # so help text that depends on the task has to find it ahead of parser run.
    # Skip the env_cfg load when --help isn't requested -- normal training runs
    # don't need the variant enumeration and hydra walks the cfg later anyway.
    task_name = _peek_task()
    actual_variants = _enumerate_variants(task_name) if (task_name and _help_requested()) else None

    group = parser.add_argument_group(
        "preset selection",
        description=(
            "Select named PresetCfg alternatives at runtime. Both '--flag value' and"
            " '--flag=value' are accepted. Flag values are folded into a 'presets=<csv>'"
            " token consumed by the Hydra-decorator flow; Hydra validates names against"
            " the loaded task at resolve time."
        ),
    )
    for target in PresetTarget:
        if target is PresetTarget.DOMAIN:
            group.add_argument(
                "--presets",
                type=str,
                default=None,
                metavar="NAME[,NAME,...]",
                help=_help_text(target, actual_variants),
            )
        else:
            group.add_argument(
                f"--{target.value}",
                type=str,
                default=None,
                metavar="NAME",
                help=_help_text(target, actual_variants),
            )

    args, remaining = parser.parse_known_args()

    # Collect names in declaration order: typed first, then free-form --presets.
    names: list[str] = []
    for target in PresetTarget:
        if target is PresetTarget.DOMAIN:
            raw = args.presets
            if raw:
                names.extend(name.strip() for name in raw.split(",") if name.strip())
        else:
            value = getattr(args, target.value, None)
            if value:
                names.append(value)

    if not names:
        # No preset flags were given; still scan *remaining* for a pre-existing
        # ``presets=...`` token so we don't drop it. If absent, hydra_argv is
        # just the un-touched remainder.
        return args, list(remaining)

    # Merge with any pre-existing ``presets=...`` token already in remaining.
    kept: list[str] = []
    for token in remaining:
        if token.startswith("presets="):
            names.extend(name.strip() for name in token[len("presets=") :].split(",") if name.strip())
        else:
            kept.append(token)

    # Dedupe, preserve first-occurrence order.
    seen: set[str] = set()
    deduped = [name for name in names if not (name in seen or seen.add(name))]
    return args, [f"presets={','.join(deduped)}", *kept]


def _help_text(target: PresetTarget, actual_variants: dict[PresetTarget, set[str]] | None) -> str:
    """Argparse ``help=`` string for a typed flag.

    The string reports the variants present in the loaded task (if a task
    was discovered via ``--task=X`` in ``sys.argv``). Without a task, it
    tells the user to pass one. The registry is not consulted here -- it
    is a naming convention hint, not a help-text source.

    Args:
        target: Which typed target's help string to build.
        actual_variants: Either ``None`` (no ``--task`` was given) or a
            ``{target: set[name]}`` mapping of variants present in the
            loaded task, bucketed by target via :func:`PresetRegistry`.
            A failure during the env_cfg load or walk is not caught
            here -- it propagates naturally to the user.

    Returns:
        Single-line help text for ``add_argument(help=...)``.
    """
    label = (
        "Comma-separated preset names" if target is PresetTarget.DOMAIN else f"{target.value.capitalize()} preset name"
    )

    if actual_variants is None:
        return f"{label}. Pass `--task=X` along with `--help` to see preset variants available for that task."

    if target is PresetTarget.DOMAIN:
        # Free-form --presets accepts any name; list every variant we found.
        all_names = sorted({n for variants in actual_variants.values() for n in variants})
        if not all_names:
            return f"{label}. No preset variants in this task."
        return f"{label} (broadcast to every matching PresetCfg). Available: {', '.join(all_names)}."

    available = sorted(actual_variants.get(target, set()))
    if not available:
        return f"{label}. No {target.value} preset variants in this task."
    return f"{label}. Available: {', '.join(available)}."


def _peek_task() -> str | None:
    """Find ``--task=X`` or ``--task X`` in ``sys.argv`` without invoking argparse.

    argparse's ``--help`` short-circuits parsing, so help text that depends
    on the task must locate it before any parser ever runs. Returns the
    *last* ``--task`` value -- matching argparse's last-wins ``store``
    semantics for repeated flags.

    Malformed values are passed through verbatim: a downstream
    ``load_cfg_from_registry`` call will raise the natural "task not
    registered" error, which is the right user-facing signal.

    Returns:
        The task value if present, otherwise ``None``.
    """
    last_task: str | None = None
    # Skip the script name at sys.argv[0].
    for i in range(1, len(sys.argv)):
        token = sys.argv[i]
        if token == "--task" and i + 1 < len(sys.argv):
            last_task = sys.argv[i + 1]
        elif token.startswith("--task="):
            last_task = token[len("--task=") :]
    return last_task


def _help_requested() -> bool:
    """Return True if ``--help`` or ``-h`` appears in ``sys.argv`` (excluding ``sys.argv[0]``)."""
    return any(token in ("--help", "-h") for token in sys.argv[1:])


def _enumerate_variants(task_name: str) -> dict[PresetTarget, set[str]]:
    """Load env_cfg for *task_name* and bucket its variants by target.

    Uses :func:`isaaclab_tasks.utils.hydra.collect_presets` -- the same
    walker hydra's resolver runs -- so help and resolve see the same view
    of the cfg tree. Env_cfg loads here are safe before ``AppLauncher``
    boots because ``test_env_cfg_no_forbidden_imports`` enforces that
    cfg modules do not import Kit-only packages at top level.

    Exceptions from :func:`load_cfg_from_registry` or :func:`collect_presets`
    propagate verbatim -- a bad ``--task`` value or a broken cfg should
    surface as the natural error the loader emits, not a string buried in
    ``--help`` text.

    Args:
        task_name: Gym registry id, e.g. ``"Isaac-Cartpole-v0"``.

    Returns:
        ``dict[PresetTarget, set[str]]`` -- variant names found in the
        task, bucketed by their registered target. Un-registered names
        fall into :attr:`PresetTarget.DOMAIN`.
    """
    from isaaclab_tasks.utils.hydra import collect_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    return _bucket_variants_by_target(collect_presets(env_cfg))


def _bucket_variants_by_target(walked: dict) -> dict[PresetTarget, set[str]]:
    """Convert :func:`collect_presets` output into ``{target: set[name]}`` by
    cfg instance type.

    For each ``(name, cfg)`` pair, the target is decided by whether
    ``type(cfg)`` matches any class registered under a typed target via
    :func:`isinstance`. Subclasses of registered classes route to their
    parent's target. Cfgs whose type matches no registered target fall
    into :attr:`PresetTarget.DOMAIN`. The implicit ``default`` field is
    filtered out -- it's the fallback, not a selectable variant the
    user can name.

    Routing by class type (not by name string) keeps target assignment
    consistent even if a task-local preset happens to reuse a backend's
    canonical name.

    Args:
        walked: Output of :func:`isaaclab_tasks.utils.hydra.collect_presets`,
            shaped as ``{path: {name: cfg, ...}, ...}``.

    Returns:
        Mapping with one entry per :class:`PresetTarget` member.
    """
    typed_targets = [t for t in PresetTarget if t is not PresetTarget.DOMAIN]
    target_classes = {t: PresetRegistry.classes_for(t) for t in typed_targets}
    result: dict[PresetTarget, set[str]] = {target: set() for target in PresetTarget}
    for path_dict in walked.values():
        for name, cfg in path_dict.items():
            if name == "default":
                continue
            matched = next(
                (t for t in typed_targets if target_classes[t] and isinstance(cfg, target_classes[t])),
                PresetTarget.DOMAIN,
            )
            result[matched].add(name)
    return result
