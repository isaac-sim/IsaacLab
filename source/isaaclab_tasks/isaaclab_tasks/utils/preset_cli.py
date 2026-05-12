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

The single new capability is **discoverability**: ``--help`` lists the
canonical names registered with
:func:`isaaclab.utils.preset_registry.register` so users learn the
typical vocabulary without typing-and-failing. The registry is a
**hint** in help text, not a CLI-time constraint.

Most scripts use the one-line form::

    parser = argparse.ArgumentParser(...)
    # ... script-specific args ...
    args_cli = setup_cli(parser)
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.utils.preset_registry import PresetRegistry, PresetTarget


def setup_cli(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Add typed preset flags + AppLauncher flags, parse, fold into ``presets=<csv>``.

    Steps:

    1. Register one argparse flag per :class:`PresetTarget`:
       ``--{target.value}=NAME`` for typed targets, ``--presets=NAME[,NAME,...]``
       for the DOMAIN catch-all. ``--help`` strings include the
       currently registered canonical names as a hint.
    2. Register AppLauncher flags via ``AppLauncher.add_app_launcher_args``.
    3. Call ``parser.parse_known_args``.
    4. Pass typed and free-form values through verbatim. Fold them
       (plus any pre-existing ``presets=...`` token in *remaining*) into
       a single ``presets=<csv>`` token; rewrite ``sys.argv`` so the
       downstream Hydra layer sees one token followed by leftover args.

    Args:
        parser: The caller's argument parser. Preset flags and the
            AppLauncher flag set are added in place; callers do NOT
            register either themselves.

    Returns:
        The namespace from ``parse_known_args``. ``sys.argv`` is
        mutated in place to carry the folded ``presets=<csv>`` token.
    """
    from isaaclab.app import AppLauncher

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
                help="Comma-separated preset names (broadcast to every matching PresetCfg).",
            )
        else:
            group.add_argument(
                f"--{target.value}",
                type=str,
                default=None,
                metavar="NAME",
                help=_help_text(target),
            )

    AppLauncher.add_app_launcher_args(parser)
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
        sys.argv = [sys.argv[0], *remaining]
        return args

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
    sys.argv = [sys.argv[0], f"presets={','.join(deduped)}", *kept]
    return args


def _help_text(target: PresetTarget) -> str:
    """Argparse ``help=`` string for a typed flag.

    Lists the canonical names currently registered for *target* as a
    hint -- the CLI doesn't gate on them, so the wording makes clear
    that any name is accepted and the resolver does the real validation.

    Args:
        target: Which typed target's help string to build.

    Returns:
        Single-line help text for ``add_argument(help=...)``.
    """
    capitalized = target.value.capitalize()
    registered = sorted(PresetRegistry.names_for(target))
    if not registered:
        return (
            f"{capitalized} preset name. Any name is accepted at the CLI; the resolver"
            " validates against the loaded task."
        )
    return (
        f"{capitalized} preset name. Common names (registered backends):"
        f" {', '.join(registered)}. Other names are accepted; the resolver validates"
        " against the loaded task."
    )
