# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed-preset selection via Hydra-style CLI tokens.

Recognizes three ``key=value`` tokens (no leading dashes) on ``sys.argv``:

* ``physics=NAME``            -- typed selector for ``PhysicsCfg`` variants.
* ``renderer=NAME``           -- typed selector for ``RendererCfg`` variants.
* ``presets=NAME[,NAME,...]`` -- broadcast applied to every matching ``PresetCfg``.

Two responsibilities, split across two functions:

* :func:`setup_preset_cli` -- register the preset-selection help description on
  the parser and run ``parse_known_args``. Returns the raw pre-fold remainder.
* :func:`fold_preset_tokens` -- rewrite typed selectors and any free-form
  ``presets=...`` tokens into a single ``presets=<csv>`` token that hydra's
  :func:`~isaaclab_tasks.utils.hydra.resolve_presets` consumes. The resolver,
  alias rewriting, and unknown-name errors are unchanged.

Splitting the fold out lets callers intersect the pre-fold remainder with
external sources (e.g. ``rsl_rl`` scripts' ``--external_callback`` hook, which
reads the user's unmutated ``sys.argv`` and returns pre-fold tokens) in the
same vocabulary. The fold runs exactly once, at the caller's final
``sys.argv`` assignment.

No argparse arguments are registered for the typed selectors -- discoverability
lives in the ``argument_group`` description, so the parsed Namespace gains no
preset attributes and cannot shadow :class:`~isaaclab.app.AppLauncher`
SimulationApp config keys (``renderer`` notably).

Typical script setup::

    parser = argparse.ArgumentParser(...)
    # ... script-specific args ...
    add_launcher_args(parser)
    args_cli, remaining = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + fold_preset_tokens(remaining)

Scripts that need to intersect the remainder with external-callback output do
the intersection first (both sides pre-fold, vocabulary matches), then fold::

    args_cli, remaining = setup_preset_cli(parser)
    if args_cli.external_callback:
        cb_remainder = external_callback_function()
        remaining = list_intersection(remaining, cb_remainder)
    sys.argv = [sys.argv[0]] + fold_preset_tokens(remaining)

``setup_preset_cli`` does NOT add AppLauncher flags itself -- callers add them
explicitly via :func:`isaaclab.app.add_launcher_args` before calling.
"""

from __future__ import annotations

import argparse
import sys

from .preset_target import PresetTarget

# ============================================================================
# Public entry point
# ============================================================================


def setup_preset_cli(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> tuple[argparse.Namespace, list[str]]:
    """Register the preset-selection help description and parse argv.

    Must be called *after* AppLauncher flags and script-specific arguments are
    registered on ``parser`` -- otherwise those unknown tokens land in
    ``parse_known_args``'s remainder.

    Does NOT fold typed selectors. The returned remainder still contains the
    user-typed ``physics=`` / ``renderer=`` / ``presets=`` tokens verbatim,
    alongside any Hydra path overrides and any unknown argparse flags. Call
    :func:`fold_preset_tokens` on the remainder before assigning ``sys.argv``;
    keeping parse and fold separate lets callers run other filters (notably
    ``rsl_rl``'s ``--external_callback`` intersection) on the pre-fold list,
    where vocabularies match.

    Does not mutate ``sys.argv``; the caller assigns
    ``sys.argv = [sys.argv[0]] + fold_preset_tokens(remaining)`` when ready, so
    any argv-aware logic that re-reads ``sys.argv`` (e.g. an external callback)
    runs against the user's original command line.

    Args:
        parser: Caller's argument parser. An ``argument_group`` is attached
            for help-time variant discovery; no ``add_argument`` calls are
            made, so the Namespace gains no preset attributes.
        argv: Optional argument list to parse. When ``None`` (default),
            ``parse_known_args`` reads from ``sys.argv``. Provided primarily
            for in-process test paths that drive the parser with a synthetic
            argv. Help-time variant enumeration always reads ``sys.argv`` --
            the user's interactive command line is the only argv that
            triggers ``--help`` rendering.

    Returns:
        ``(args, remaining)`` where ``remaining`` is the verbatim output of
        ``parser.parse_known_args(argv)``. Apply :func:`fold_preset_tokens`
        to ``remaining`` before handing it to Hydra.
    """
    # --help short-circuits parsing, so help text that depends on --task has to
    # find it before argparse runs. Gate the env_cfg load on --help to keep
    # normal training runs cheap.
    argv_helper = _ArgvHelper(sys.argv)
    actual_variants = (
        _enumerate_variants(argv_helper.task_name) if (argv_helper.task_name and argv_helper.help_requested) else None
    )

    # Argparse's default HelpFormatter reflows description text into one wrapped
    # paragraph, which would collapse the per-variant bullets we emit. Use a
    # formatter that wraps each blank-line-separated paragraph independently
    # while preserving explicit newlines. Respect a caller-set custom formatter.
    if parser.formatter_class is argparse.HelpFormatter:
        parser.formatter_class = _PresetHelpFormatter

    # Help-only group: no add_argument() calls means no preset attributes on
    # the Namespace, so AppLauncher can't accidentally forward one (notably
    # ``renderer``) into SimulationApp config.
    parser.add_argument_group("preset selection", description=_DescriptionBuilder.build(actual_variants))

    return parser.parse_known_args(argv)


def fold_preset_tokens(tokens: list[str]) -> list[str]:
    """Fold preset selector tokens into a single ``presets=<csv>`` token.

    Recognises ``physics=NAME`` / ``renderer=NAME`` / ``presets=NAME[,NAME,...]``
    in *tokens* (exact key match; dotted keys like ``env.sim.physics=NAME`` are
    path-targeted overrides and pass through unchanged). All recognised names
    are deduped in first-occurrence order and emitted as a leading
    ``presets=<csv>`` token; every other token in *tokens* is appended in its
    original position.

    Call this on the remainder returned by :func:`setup_preset_cli` before
    assigning ``sys.argv``. Scripts that intersect the remainder with
    callback-returned tokens (e.g. ``rsl_rl/{train,play}.py``'s
    ``--external_callback`` flow) must do the intersection *first* (both sides
    pre-fold) and then call this function.

    Args:
        tokens: Pre-fold token list (typically the second element of the
            tuple returned by :func:`setup_preset_cli`).

    Returns:
        A new list with selector tokens folded into one leading
        ``presets=<csv>`` token if any were present; otherwise the input list
        is returned unchanged.
    """
    typed_labels = {t.value for t in PresetTarget if t.base_classes}
    names: list[str] = []
    kept: list[str] = []
    for token in tokens:
        if "=" not in token:
            kept.append(token)
            continue
        key, val = token.split("=", 1)
        if key in typed_labels:
            # Typed selector value is a single name; commas are reserved for ``presets=`` broadcast.
            stripped = val.strip()
            if stripped:
                names.append(stripped)
        elif key == PresetTarget.DOMAIN.value:
            names.extend(name.strip() for name in val.split(",") if name.strip())
        else:
            kept.append(token)

    if not names:
        return list(kept)

    # Dedupe, preserve first-occurrence order.
    seen: set[str] = set()
    deduped = [name for name in names if not (name in seen or seen.add(name))]
    return [f"presets={','.join(deduped)}", *kept]


# ============================================================================
# Public preset enumeration (for tooling, e.g. list_envs)
# ============================================================================


def enumerate_task_presets(task_name: str) -> dict[PresetTarget, list[str]] | None:
    """Return the available preset names for *task_name*, bucketed by selector type.

    Loads the env config registered under *task_name* and walks its preset tree
    using the same logic that the CLI help-text renderer uses, so the returned
    view matches what ``--task=<name> --help`` shows at the command line.

    This function is safe to call after :class:`~isaaclab.app.AppLauncher` has
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
        result = _enumerate_variants(task_name)
        return {target: sorted(names) for target, names in result.items()}
    except Exception:
        return None


# ============================================================================
# Help-text rendering
# ============================================================================


class _PresetHelpFormatter(argparse.HelpFormatter):
    """Argparse help formatter that wraps each paragraph separately.

    Default :class:`argparse.HelpFormatter` reflows the entire description into
    one paragraph, merging the variant listing into the surrounding prose, and
    collapses ``\\n``-separated bullets onto one line.
    :class:`~argparse.RawDescriptionHelpFormatter` preserves description
    newlines but drops wrapping entirely. The ``_fill_text`` override below
    splits the description on blank lines and wraps each paragraph indep-
    endently, giving both readable paragraphs and per-line bullets.
    """

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        import textwrap

        paragraphs = text.split("\n\n")
        rendered: list[str] = []
        for paragraph in paragraphs:
            # A paragraph that already contains hard newlines (the bulleted
            # variant listing) is rendered verbatim; otherwise word-wrap.
            if "\n" in paragraph:
                rendered.append("\n".join(f"{indent}{line}" for line in paragraph.splitlines()))
            else:
                rendered.append(textwrap.fill(paragraph, width, initial_indent=indent, subsequent_indent=indent))
        return "\n\n".join(rendered)


class _DescriptionBuilder:
    """Renders the preset-selection ``argument_group`` description.

    Groups the column constants and per-row formatting that build the
    selector table. Iterates :class:`PresetTarget` to produce one row per
    selector; each row's syntax and description come from the enum, so
    adding a new typed target needs no changes here.
    """

    # Column widths. ``SELECTOR_COL`` = width of the longest selector syntax
    # (``presets=NAME[,NAME,...]`` = 23 chars); shorter selectors right-pad
    # to this width. ``DESC_GAP`` is the gap between syntax and description.
    SELECTOR_COL = 23
    DESC_GAP = 3
    ROW_PREFIX = "    "

    INTRO = "Select named PresetCfg alternatives via Hydra-style overrides (key=value, no leading dashes):"
    EPILOG = "Hydra also accepts path-targeted overrides like env.sim.physics=NAME."
    HINT = "Pass `--task=X` along with `--help` to see preset variants available for that task."

    @classmethod
    def build(cls, actual_variants: dict[PresetTarget, set[str]] | None) -> str:
        """Build the description text.

        Args:
            actual_variants: ``None`` when no ``--task=X --help`` is in argv;
                otherwise a ``{target: set[name]}`` bucketed view from
                :func:`_enumerate_variants`.
        """
        with_available = actual_variants is not None
        rows = [
            cls._row(t, with_available=with_available, variants=sorted((actual_variants or {}).get(t, set())))
            for t in PresetTarget
        ]
        middle = f"{cls.HINT}\n\n" if not with_available else ""
        return f"{cls.INTRO}\n" + "\n".join(rows) + f"\n\n{middle}{cls.EPILOG}"

    @classmethod
    def _row(cls, target: PresetTarget, *, with_available: bool, variants: list[str]) -> str:
        syntax = cls._syntax(target).ljust(cls.SELECTOR_COL)
        desc = cls._description(target)
        suffix = ". Available:" if with_available else ""
        header = f"{cls.ROW_PREFIX}{syntax}{' ' * cls.DESC_GAP}{desc}{suffix}"
        if not with_available:
            return header
        # Bullet indent aligns with the description column once argparse
        # prepends its 2-space group-description indent.
        bullet_indent = " " * (len(cls.ROW_PREFIX) + cls.SELECTOR_COL + cls.DESC_GAP)
        body = "\n".join(f"{bullet_indent}- {n}" for n in variants) if variants else f"{bullet_indent}(none)"
        return f"{header}\n{body}"

    @staticmethod
    def _syntax(target: PresetTarget) -> str:
        """User-facing selector form: ``physics=NAME`` vs ``presets=NAME[,NAME,...]``."""
        if target.base_classes:  # typed: single name
            return f"{target.value}=NAME"
        return f"{target.value}=NAME[,NAME,...]"  # DOMAIN: comma-separated broadcast

    @staticmethod
    def _description(target: PresetTarget) -> str:
        """One-line description; for typed targets includes the cfg base class name."""
        if target.base_classes:
            return f"(typed) selects a {target.base_classes[0].__name__} variant"
        return "broadcast: applied to every matching PresetCfg"


# ============================================================================
# argv inspection (pre-argparse peek for help-text rendering)
# ============================================================================


class _ArgvHelper:
    """Single-pass argv scan that exposes ``task_name`` and ``help_requested``.

    Needed because argparse's ``--help`` short-circuits parsing, so help text
    that depends on ``--task`` has to find it before argparse runs.

    Attributes:
        task_name: Last ``--task`` value (matching argparse's last-wins
            semantics), or ``None`` if absent.
        help_requested: ``True`` if ``--help`` or ``-h`` is present.
    """

    def __init__(self, argv: list[str]):
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

    Uses the same walker hydra's resolver runs so help and resolve see one
    view of the cfg tree. The env_cfg load is safe before AppLauncher boots
    because ``test_env_cfg_no_forbidden_imports`` blocks Kit-only imports at
    the top level of cfg modules. Exceptions from the loader propagate
    verbatim -- they surface as the natural error, not a buried help string.
    """
    from isaaclab_tasks.utils.hydra import collect_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    return _bucket_variants_by_target(collect_presets(env_cfg))


def _bucket_variants_by_target(walked: dict) -> dict[PresetTarget, set[str]]:
    """Convert :func:`collect_presets` output into ``{target: set[name]}``.

    Routes each ``(name, cfg)`` by ``isinstance(cfg, target.base_classes)``;
    cfgs matching no typed target fall into ``DOMAIN``. The implicit
    ``default`` field is filtered -- it's the fallback, not a selectable name.

    Routing by class hierarchy means new backends subclassing
    :class:`~isaaclab.physics.PhysicsCfg` /
    :class:`~isaaclab.renderers.renderer_cfg.RendererCfg` bucket automatically
    regardless of what name the env_cfg gives the field.
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
