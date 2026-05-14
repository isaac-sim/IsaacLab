# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed-preset selection via Hydra-style CLI tokens.

Recognizes three ``key=value`` tokens (no leading dashes) on ``sys.argv``:

* ``physics=NAME``            -- typed selector for ``PhysicsCfg`` variants.
* ``renderer=NAME``           -- typed selector for ``RendererCfg`` variants.
* ``presets=NAME[,NAME,...]`` -- broadcast applied to every matching ``PresetCfg``.

All three fold into a single ``presets=<csv>`` token that hydra's
:func:`~isaaclab_tasks.utils.hydra.resolve_presets` already consumes; the
resolver, alias rewriting, and unknown-name errors are unchanged.

The grammar matches Hydra's, so a single ``sys.argv`` carries both typed
selectors and path-targeted overrides (``env.scene.num_envs=4096``); these
flow through ``parse_known_args`` into ``remaining`` and are handled here.

No argparse arguments are registered for the typed selectors -- discoverability
lives in the ``argument_group`` description, so the parsed Namespace gains no
preset attributes and cannot shadow :class:`~isaaclab.app.AppLauncher`
SimulationApp config keys (``renderer`` notably).

Typical script setup::

    parser = argparse.ArgumentParser(...)
    # ... script-specific args ...
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

``setup_preset_cli`` does NOT add AppLauncher flags itself -- callers add them
explicitly via :func:`isaaclab_tasks.utils.add_launcher_args` before calling.
"""

from __future__ import annotations

import argparse
import sys

from .preset_target import PresetTarget

# ============================================================================
# Public entry point
# ============================================================================


def setup_preset_cli(parser: argparse.ArgumentParser) -> tuple[argparse.Namespace, list[str]]:
    """Render typed-preset help, parse, and fold typed selectors into a Hydra token.

    Must be called *after* AppLauncher flags and script-specific arguments are
    registered on ``parser`` -- otherwise those unknown tokens land in
    ``parse_known_args``'s remainder.

    Does not mutate ``sys.argv``; the caller assigns
    ``sys.argv = [sys.argv[0]] + hydra_argv`` when ready, so any argv-aware
    logic (e.g. an ``--external_callback`` hook that re-reads ``sys.argv``)
    runs against the user's original command line first.

    Args:
        parser: Caller's argument parser. An ``argument_group`` is attached
            for help-time variant discovery; no ``add_argument`` calls are
            made, so the Namespace gains no preset attributes.

    Returns:
        ``(args, hydra_argv)``. ``hydra_argv[0]`` is a folded ``presets=<csv>``
        token whenever any typed selector or free-form ``presets=...`` was
        present; otherwise the list is the non-preset remainder only.
    """
    # --help short-circuits parsing, so help text that depends on --task has to
    # find it before argparse runs. Gate the env_cfg load on --help to keep
    # normal training runs cheap.
    argv = _ArgvHelper(sys.argv)
    actual_variants = _enumerate_variants(argv.task_name) if (argv.task_name and argv.help_requested) else None

    # Argparse's default HelpFormatter reflows description text into one wrapped
    # paragraph, which would collapse the per-variant bullets we emit. Use a
    # formatter that wraps each blank-line-separated paragraph independently
    # while preserving explicit newlines. Respect a caller-set custom formatter.
    if parser.formatter_class is argparse.HelpFormatter:
        parser.formatter_class = _PresetHelpFormatter

    # Help-only group: no add_argument() calls means no preset attributes on
    # the Namespace, so AppLauncher can't accidentally forward one (notably
    # ``renderer``) into SimulationApp config.
    parser.add_argument_group("preset selection", description=_build_description(actual_variants))

    args, remaining = parser.parse_known_args()

    typed_labels = {t.value for t in PresetTarget if t is not PresetTarget.DOMAIN}
    names: list[str] = []
    kept: list[str] = []
    for token in remaining:
        if "=" not in token:
            kept.append(token)
            continue
        key, val = token.split("=", 1)
        if key in typed_labels:
            # Typed selector value is a single name; commas are reserved for ``presets=`` broadcast.
            stripped = val.strip()
            if stripped:
                names.append(stripped)
        elif key == "presets":
            names.extend(name.strip() for name in val.split(",") if name.strip())
        else:
            kept.append(token)

    if not names:
        return args, kept

    # Dedupe, preserve first-occurrence order.
    seen: set[str] = set()
    deduped = [name for name in names if not (name in seen or seen.add(name))]
    return args, [f"presets={','.join(deduped)}", *kept]


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


def _build_description(actual_variants: dict[PresetTarget, set[str]] | None) -> str:
    """Build the preset-selection ``argument_group`` description.

    Renders each selector entry inline with its available variants directly
    underneath, so users see the bucketed variants at the point where they
    decide which selector to use. Emits ``\\n``-separated lines; the caller
    sets the parser's ``formatter_class`` to one that preserves them.

    Args:
        actual_variants: ``None`` when no ``--task=X --help`` is in argv;
            otherwise a ``{target: set[name]}`` bucketed view from
            :func:`_enumerate_variants`.
    """
    intro = "Select named PresetCfg alternatives via Hydra-style overrides (key=value, no leading dashes):"
    epilog = "Hydra also accepts path-targeted overrides like env.sim.physics=NAME."

    if actual_variants is None:
        # No task yet -- just describe each selector; variants are task-
        # specific and listed only when --task=X is also given.
        selector_block = (
            "    physics=NAME              (typed) selects a PhysicsCfg variant\n"
            "    renderer=NAME             (typed) selects a RendererCfg variant\n"
            "    presets=NAME[,NAME,...]   broadcast: applied to every matching PresetCfg"
        )
        hint = "Pass `--task=X` along with `--help` to see preset variants available for that task."
        return f"{intro}\n{selector_block}\n\n{hint}\n\n{epilog}"

    # 30 spaces -> bullets land at column 33 once argparse prepends its 2-space
    # group-description indent, aligning with where the inline ``(typed)`` /
    # ``broadcast:`` descriptions begin under each selector header.
    bullet_indent = " " * 30

    def _variant_lines(names: list[str]) -> str:
        return "\n".join(f"{bullet_indent}- {n}" for n in names) if names else f"{bullet_indent}(none)"

    physics = sorted(actual_variants.get(PresetTarget.PHYSICS, set()))
    renderer = sorted(actual_variants.get(PresetTarget.RENDERER, set()))
    # DOMAIN is the catch-all: variants whose cfg type doesn't subclass any typed
    # target's base. Typed variants stay only under their typed selector and are
    # intentionally not duplicated under ``presets=`` (which would be
    # broadcast-applied if used).
    domain = sorted(actual_variants.get(PresetTarget.DOMAIN, set()))

    phys_header = "physics=NAME              (typed) selects a PhysicsCfg variant. Available:"
    rend_header = "renderer=NAME             (typed) selects a RendererCfg variant. Available:"
    dom_header = "presets=NAME[,NAME,...]   broadcast: applied to every matching PresetCfg. Available:"
    body = (
        f"    {phys_header}\n{_variant_lines(physics)}\n"
        f"    {rend_header}\n{_variant_lines(renderer)}\n"
        f"    {dom_header}\n{_variant_lines(domain)}"
    )
    return f"{intro}\n{body}\n\n{epilog}"


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
