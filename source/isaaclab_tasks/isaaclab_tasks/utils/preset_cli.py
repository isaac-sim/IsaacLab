# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed-preset selection via Hydra-style CLI tokens.

Recognizes three ``key=value`` tokens (no leading dashes) on ``sys.argv``:

* ``physics=NAME``            -- typed selector for ``PhysicsCfg`` variants.
* ``renderer=NAME``           -- typed selector for ``RendererCfg`` variants.
* ``presets=NAME[,NAME,...]`` -- broadcast applied to every matching ``PresetCfg``.

:func:`setup_preset_cli` registers preset-selection help, then runs
``parse_known_args`` and returns the verbatim remainder. The preset tokens
above are passed through unchanged; hydra's
:func:`~isaaclab_tasks.utils.hydra.register_task` parses them directly (applying
the names as presets and enforcing that ``physics=``/``renderer=`` resolve
against a config of that type). Callers simply assign the remainder to
``sys.argv``; no rewriting step is needed.

No argparse arguments are registered for the typed selectors -- their
discoverability lives in the ``argument_group`` description, so the parsed
Namespace gains no preset attributes and cannot shadow
:class:`~isaaclab_physx.app.KitLauncher` SimulationApp config keys (``renderer``
notably).

Typical script setup::

    parser = argparse.ArgumentParser(...)
    # ... script-specific args ...
    add_launcher_args(parser)
    args_cli, remaining = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + remaining

Scripts that intersect the remainder with external-callback output (e.g.
``rsl_rl`` scripts' ``--external_callback`` hook) do the intersection on the
remainder before assigning ``sys.argv`` -- both sides share the same token
vocabulary::

    args_cli, remaining = setup_preset_cli(parser)
    if args_cli.external_callback:
        remaining = list_intersection(remaining, external_callback_function())
    sys.argv = [sys.argv[0]] + remaining

``setup_preset_cli`` does NOT add launcher flags itself -- callers add them
explicitly via :func:`isaaclab.app.add_launcher_args` before calling.
"""

from __future__ import annotations

import argparse
import sys

from .hydra import collect_presets
from .preset_target import PresetTarget

# ============================================================================
# Public entry point
# ============================================================================


def setup_preset_cli(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> tuple[argparse.Namespace, list[str]]:
    """Register the preset-selection help description and parse argv.

    Must be called *after* launcher flags and script-specific arguments are
    registered on ``parser`` -- otherwise those unknown tokens land in
    ``parse_known_args``'s remainder.

    The returned remainder contains the user-typed ``physics=`` / ``renderer=``
    / ``presets=`` tokens verbatim, alongside any Hydra path overrides and any
    unknown argparse flags, ready to assign to ``sys.argv`` for hydra to parse.

    Does not mutate ``sys.argv``; the caller assigns
    ``sys.argv = [sys.argv[0]] + remaining`` when ready, so any argv-aware logic
    that re-reads ``sys.argv`` (e.g. an external callback) runs against the
    user's original command line first.

    Args:
        parser: Caller's argument parser. An ``argument_group`` is attached
            for help-time variant discovery. No preset selector arguments are
            added, so the Namespace gains no preset attributes.
        argv: Arguments to parse, including help-time task selection. Defaults to ``sys.argv[1:]``.
    A Hydra override is never the value of an option whose value is optional
    (``nargs="?"``): ``--video presets=newton_mjwarp`` records video with the
    option's default source and passes the override on. Overrides always
    contain ``=``, so such an option followed by a token with ``=`` takes its
    ``const`` instead.

    Returns:
        ``(args, remaining)`` where ``remaining`` is the verbatim output of
        ``parser.parse_known_args(argv)``, ready to hand to Hydra via
        ``sys.argv``.

    Raises:
        SystemExit: If ``argv`` requests help, after printing it.
    """
    args_to_parse = sys.argv[1:] if argv is None else argv
    help_requested = "-h" in args_to_parse or "--help" in args_to_parse
    variants = None
    if help_requested:
        # Read the task before the main parser's help action exits.
        task_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        task_parser.add_argument("--task")
        task_args, _ = task_parser.parse_known_args(args_to_parse)
        if task_args.task:
            variants = _enumerate_variants(task_args.task)
    if parser.formatter_class is argparse.HelpFormatter:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
    description = ["Select named PresetCfg alternatives via Hydra-style overrides (key=value, no leading dashes):"]
    for target in PresetTarget:
        syntax = f"{target.value}=NAME"
        if target is PresetTarget.DOMAIN:
            syntax += "[,NAME,...]"
        meaning = (
            f"(typed) selects a {target.value} backend"
            if target.base_classes
            else "broadcast: applied to every matching PresetCfg"
        )
        description.append(f"    {syntax:23}   {meaning}" + (". Available:" if variants is not None else ""))
        if variants is not None:
            description.extend(f"                              - {name}" for name in variants[target])
            if not variants[target]:
                description.append("                              (none)")
    if variants is None:
        description.append("\nPass `--task=X` along with `--help` to see preset variants available for that task.")
    description.append("\nHydra also accepts path-targeted overrides like env.sim.physics=NAME.")
    parser.add_argument_group("preset selection", description="\n".join(description))
    if help_requested:
        parser.print_help()
        raise SystemExit(0)

    optional_value_consts = {
        option: action.const
        for action in parser._actions
        if action.nargs == argparse.OPTIONAL
        for option in action.option_strings
    }
    args_to_parse = [
        f"{token}={optional_value_consts[token]}" if token in optional_value_consts and "=" in next_token else token
        for token, next_token in zip(args_to_parse, [*args_to_parse[1:], ""])
    ]
    return parser.parse_known_args(args_to_parse)


# ============================================================================
# Public preset enumeration (for tooling, e.g. list_envs)
# ============================================================================


def enumerate_task_presets(task_name: str) -> dict[PresetTarget, list[str]] | None:
    """Return the available preset names for *task_name*, bucketed by selector type.

    Loads the env config registered under *task_name* and walks its preset tree
    using the same logic that the CLI help-text renderer uses, so the returned
    view matches what ``--task=<name> --help`` shows at the command line.

    This function is safe to call after :class:`~isaaclab_physx.app.KitLauncher` has
    booted (i.e. inside a running Isaac Sim session).

    Args:
        task_name: Gymnasium task ID (e.g. ``"Isaac-Cartpole"``).

    Returns:
        A mapping ``{PresetTarget: sorted list of preset names}`` on success.
        Returns ``None`` if the env config cannot be loaded (import error,
        missing registration, etc.).  The ``"default"`` fallback is excluded
        from every list because it is implicit, not a user-selectable name.
    """
    try:
        return _enumerate_variants(task_name)
    except Exception:
        return None


def _enumerate_variants(task_name: str) -> dict[PresetTarget, list[str]]:
    """Load env_cfg for *task_name* and bucket its variants by target.

    Uses the same walker hydra's resolver runs so help and resolve see one
    view of the cfg tree. The env_cfg load is safe before the simulation runtime starts
    because ``test_env_cfg_no_forbidden_imports`` blocks Kit-only imports at
    the top level of cfg modules. Exceptions from the loader propagate
    verbatim -- they surface as the natural error, not a buried help string.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    typed_targets = [t for t in PresetTarget if t.base_classes]
    result: dict[PresetTarget, set[str]] = {target: set() for target in PresetTarget}
    for path_dict in collect_presets(env_cfg).values():
        for name, cfg in path_dict.items():
            if name == "default":
                continue
            matched = next(
                (target for target in typed_targets if target.matches(cfg)),
                PresetTarget.DOMAIN,
            )
            result[matched].add(name)
    return {target: sorted(names) for target, names in result.items()}
