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
  is unknown might still be a legitimate task-local preset
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
:func:`isaaclab_tasks.utils.hydra.collect_presets`, and buckets the
variants by ``isinstance`` against each
:attr:`PresetTarget.base_classes` so typed flags
(``--physics`` / ``--renderer``) list only their own kind. The
load is safe before ``AppLauncher`` boots because IsaacLab's
``test_env_cfg_no_forbidden_imports.py`` enforces that env_cfg modules
do not import ``pxr`` / ``omni`` / ``carb`` / ``isaacsim`` at top
level. Without ``--task``, ``--help`` tells the user to pass one
(the available variants are task-dependent and we don't try to guess).

Typical script setup::

    parser = argparse.ArgumentParser(...)
    # ... script-specific args ...
    add_launcher_args(parser)  # AppLauncher flags (--headless, --device, ...)
    args_cli, hydra_args = setup_preset_cli(parser)  # adds preset flags + parses
    sys.argv = [sys.argv[0]] + hydra_args  # caller hands the folded argv to Hydra

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

from .preset_target import PresetTarget

# ============================================================================
# Public entry point
# ============================================================================


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
    # Single pre-argparse scan of sys.argv: --help short-circuits parsing, so any
    # help text that depends on --task has to find it before argparse runs.
    # Variant enumeration is gated on --help being requested -- normal training
    # runs skip the env_cfg load entirely (hydra walks the cfg later anyway).
    argv = _ArgvHelper(sys.argv)
    actual_variants = _enumerate_variants(argv.task_name) if (argv.task_name and argv.help_requested) else None

    description = (
        "Select named PresetCfg alternatives at runtime. Both '--flag value' and"
        " '--flag=value' are accepted. Flag values are folded into a 'presets=<csv>'"
        " token consumed by the Hydra-decorator flow; Hydra validates names against"
        " the loaded task at resolve time."
    )
    if actual_variants is None:
        # Hoist the "no task yet" hint to the section header so it prints once,
        # instead of repeating identical text in each typed flag's help string.
        # The "\n\n" puts the hint on its own paragraph so it stands out from
        # the surrounding description blurb.
        description += "\n\nPass `--task=X` along with `--help` to see preset variants available for that task."
    # Default formatter reflows argument help into wrapped paragraphs, which
    # would collapse the per-variant bullets emitted by ``_help_text`` into one
    # line. Switch to a formatter that honors ``\n`` in argument help while
    # still wrapping unmarked text -- AppLauncher's own help strings have no
    # explicit newlines so they keep wrapping. Respect a caller-set formatter.
    if parser.formatter_class is argparse.HelpFormatter:
        parser.formatter_class = _PresetHelpFormatter
    group = parser.add_argument_group("preset selection", description=description)
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

    # Pop the preset-flag values off the namespace as we collect them. Leaving
    # them on ``args`` would let AppLauncher's name-based forwarding pick up,
    # e.g., ``args.renderer`` and push it into ``SimulationApp.config["renderer"]``
    # (which then crashes on ``None.lower()``). After this loop the namespace
    # carries no preset attributes; the values live only in ``hydra_argv``.
    ns = vars(args)
    names: list[str] = []
    for target in PresetTarget:
        if target is PresetTarget.DOMAIN:
            raw = ns.pop("presets", None)
            if raw:
                names.extend(name.strip() for name in raw.split(",") if name.strip())
        else:
            value = ns.pop(target.value, None)
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


# ============================================================================
# Help-text rendering
# ============================================================================


class _PresetHelpFormatter(argparse.HelpFormatter):
    """Argparse help formatter that wraps each paragraph separately and
    preserves ``\\n`` inside argument help strings.

    Default :class:`argparse.HelpFormatter` reflows the entire description
    into one paragraph, merging the ``Pass --task=X`` hint into the surrounding
    prose, and collapses the per-variant bullets emitted by :func:`_help_text`
    into one line. :class:`~argparse.RawDescriptionHelpFormatter` solves the
    first issue but drops wrapping entirely, leaving descriptions as one long
    line. The overrides below do both: ``_fill_text`` wraps each
    blank-line-separated paragraph independently, and ``_split_lines`` honors
    explicit newlines in argument help. Arguments without ``\\n``
    (AppLauncher's flags) wrap exactly as before.
    """

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        import textwrap

        paragraphs = text.split("\n\n")
        return "\n\n".join(textwrap.fill(p, width, initial_indent=indent, subsequent_indent=indent) for p in paragraphs)

    def _split_lines(self, text: str, width: int) -> list[str]:
        if "\n" not in text:
            return super()._split_lines(text, width)
        out: list[str] = []
        for segment in text.splitlines():
            if segment == "":
                out.append("")
            else:
                out.extend(super()._split_lines(segment, width))
        return out


def _help_text(target: PresetTarget, actual_variants: dict[PresetTarget, set[str]] | None) -> str:
    """Argparse ``help=`` string for a typed flag.

    The string reports the variants present in the loaded task (if a task
    was discovered via ``--task=X`` in ``sys.argv``). Without a task,
    the per-flag string is just the label -- the "pass ``--task=X``"
    hint lives once on the section description (see
    :func:`setup_preset_cli`) so it isn't repeated three times. The
    registry is not consulted here -- it is a naming convention hint,
    not a help-text source.

    Args:
        target: Which typed target's help string to build.
        actual_variants: Either ``None`` (no ``--task`` was given) or a
            ``{target: set[name]}`` mapping of variants present in the
            loaded task, bucketed by target via ``isinstance`` against
              :attr:`PresetTarget.base_classes`.
            A failure during the env_cfg load or walk is not caught
            here -- it propagates naturally to the user.

    Returns:
        Single-line help text for ``add_argument(help=...)``.
    """
    label = (
        "Comma-separated preset names" if target is PresetTarget.DOMAIN else f"{target.value.capitalize()} preset name"
    )

    if actual_variants is None:
        return f"{label}."

    if target is PresetTarget.DOMAIN:
        # Free-form --presets accepts any name; list every variant we found.
        all_names = sorted({n for variants in actual_variants.values() for n in variants})
        prefix = f"{label} (broadcast to every matching PresetCfg)."
        if not all_names:
            return f"{prefix} No preset variants in this task."
        return f"{prefix} Available:\n" + "\n".join(f"  - {name}" for name in all_names)

    available = sorted(actual_variants.get(target, set()))
    if not available:
        return f"{label}. No {target.value} preset variants in this task."
    return f"{label}. Available:\n" + "\n".join(f"  - {name}" for name in available)


# ============================================================================
# argv inspection (pre-argparse peek for help-text rendering)
# ============================================================================


class _ArgvHelper:
    """Single-pass scan of an argv list that exposes the facts ``setup_preset_cli``
    needs before argparse runs.

    argparse's ``--help`` action short-circuits parsing, so help text that
    depends on ``--task`` has to find it before any parser ever sees the
    tokens. The same pass also tells us whether ``--help`` was requested at
    all -- a normal training run skips the env_cfg load that powers the
    task-aware help text.

    Attributes:
        task_name: The *last* ``--task`` value (matching argparse's
            last-wins ``store`` semantics for repeated flags), or
            ``None`` if absent. Malformed values are passed through
            verbatim; a downstream ``load_cfg_from_registry`` call will
            raise the natural "task not registered" error.
        help_requested: ``True`` if ``--help`` or ``-h`` appears in
            *argv* (excluding ``argv[0]``).
    """

    def __init__(self, argv: list[str]):
        """Scan *argv* once and populate :attr:`task_name` and :attr:`help_requested`.

        Args:
            argv: Argument list to inspect, typically ``sys.argv``. The
                element at index 0 (script name) is skipped.
        """
        self.task_name: str | None = None
        self.help_requested: bool = False
        for i in range(1, len(argv)):
            token = argv[i]
            if token in ("--help", "-h"):
                self.help_requested = True
            elif token == "--task" and i + 1 < len(argv):
                self.task_name = argv[i + 1]
            elif token.startswith("--task="):
                self.task_name = token[len("--task=") :]


# ============================================================================
# Help-time variant enumeration (load env_cfg, walk, bucket by target)
# ============================================================================


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
        task, bucketed by ``isinstance`` against each typed target's
        :attr:`~PresetTarget.base_classes`. Variants whose cfg type does
        not subclass any typed target's base fall into
        :attr:`PresetTarget.DOMAIN`.
    """
    from isaaclab_tasks.utils.hydra import collect_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    return _bucket_variants_by_target(collect_presets(env_cfg))


def _bucket_variants_by_target(walked: dict) -> dict[PresetTarget, set[str]]:
    """Convert :func:`collect_presets` output into ``{target: set[name]}`` by
    cfg instance type.

    For each ``(name, cfg)`` pair, the target is decided by ``isinstance(cfg,
    target.base_classes)`` against each typed target on :class:`PresetTarget`.
    The first match wins; cfgs that match no typed target's base classes fall
    into :attr:`PresetTarget.DOMAIN`. The implicit ``default`` field is
    filtered out -- it's the fallback, not a selectable variant the
    user can name.

    Routing by class hierarchy (not by name string) keeps target assignment
    consistent regardless of how an env_cfg names the PresetCfg field, and
    any new backend that subclasses :class:`~isaaclab.physics.PhysicsCfg` or
    :class:`~isaaclab.renderers.renderer_cfg.RendererCfg` is picked up
    automatically.

    Args:
        walked: Output of :func:`isaaclab_tasks.utils.hydra.collect_presets`,
            shaped as ``{path: {name: cfg, ...}, ...}``.

    Returns:
        Mapping with one entry per :class:`PresetTarget` member.
    """
    typed_targets = [t for t in PresetTarget if t.base_classes]
    result: dict[PresetTarget, set[str]] = {target: set() for target in PresetTarget}
    for path_dict in walked.values():
        for name, cfg in path_dict.items():
            if name == "default":
                continue
            matched = next(
                (t for t in typed_targets if isinstance(cfg, t.base_classes)),
                PresetTarget.DOMAIN,
            )
            result[matched].add(name)
    return result
