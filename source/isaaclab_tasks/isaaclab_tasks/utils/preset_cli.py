# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed-preset selection via Hydra-style CLI tokens.

Recognizes three ``key=value`` tokens (no leading dashes) on ``sys.argv``:

* ``physics=NAME``            -- typed selector for ``PhysicsCfg`` variants.
* ``renderer=NAME``           -- typed selector for ``RendererCfg`` variants.
* ``presets=NAME[,NAME,...]`` -- broadcast applied to every matching ``PresetCfg``.

:func:`setup_preset_cli` registers preset-selection help and, for RL callers,
agent discovery. It then runs ``parse_known_args``, returning the verbatim
remainder. The preset tokens above are passed through unchanged; hydra's
:func:`~isaaclab_tasks.utils.hydra.register_task` parses them directly (applying
the names as presets and enforcing that ``physics=``/``renderer=`` resolve
against a config of that type). Callers simply assign the remainder to
``sys.argv``; no rewriting step is needed.

No argparse arguments are registered for the typed selectors -- their
discoverability lives in the ``argument_group`` description, so the parsed
Namespace gains no preset attributes and cannot shadow
:class:`~isaaclab.app.AppLauncher` SimulationApp config keys (``renderer``
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
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
    *,
    agent_library: str | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Register the preset-selection help description and parse argv.

    Must be called *after* AppLauncher flags and script-specific arguments are
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
        argv: Optional argument list to parse. When ``None`` (default),
            ``parse_known_args`` reads from ``sys.argv``. Provided primarily
            for in-process test paths that drive the parser with a synthetic
            argv. Help-time variant enumeration always reads ``sys.argv`` --
            the user's interactive command line is the only argv that
            triggers ``--help`` rendering.
        agent_library: Optional RL-library prefix. When provided, task-specific
            help lists registered ``--agent`` values and declared preset
            compatibility, and a defaulted ``--agent`` is resolved from the
            task's registry metadata by :func:`_auto_select_agent`.

    Returns:
        ``(args, remaining)`` where ``remaining`` is the verbatim output of
        ``parser.parse_known_args(argv)``, ready to hand to Hydra via
        ``sys.argv``. ``args.agent`` may have been filled in from the task's
        registry metadata; the preset tokens that drove that choice stay in
        ``remaining`` for hydra.

    Raises:
        SystemExit: If ``argv`` requests help, after printing it.
    """
    # --help short-circuits parsing, so help text that depends on --task has to
    # find it before argparse runs. Gate the env_cfg load on --help to keep
    # normal training runs cheap.
    argv_helper = _ArgvHelper(sys.argv)
    actual_variants = None
    if argv_helper.task_name and argv_helper.help_requested:
        actual_variants = _enumerate_variants(argv_helper.task_name)

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

    if agent_library:
        parser.add_argument_group(
            "agent selection",
            description=_AgentDescriptionBuilder.build(agent_library, argv_helper.task_name),
        )

    args_to_parse = sys.argv[1:] if argv is None else argv
    if "-h" in args_to_parse or "--help" in args_to_parse:
        parser.print_help()
        raise SystemExit(0)

    args, remaining = parser.parse_known_args(args_to_parse)

    task_name = getattr(args, "task", None) or argv_helper.task_name
    if agent_library and task_name:
        _auto_select_agent(args, parser, task_name, agent_library, _ArgvHelper.from_tokens(args_to_parse))

    return args, remaining


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


class _AgentDescriptionBuilder:
    """Render registered agent configs and declared preset compatibility."""

    @staticmethod
    def build(agent_library: str, task_name: str | None) -> str:
        """Build help text for one RL library.

        Args:
            agent_library: RL-library prefix used to filter agent configs.
            task_name: Gymnasium task ID, or ``None`` when task-specific help
                was not requested.

        Returns:
            Multi-line argparse group description.
        """
        if task_name is None:
            return (
                f"Registered --agent values for {agent_library}. Pass `--task=X --help` "
                "to see the available configs and declared preset compatibility."
            )

        agents, compatibility = _enumerate_agents(task_name, agent_library)
        if not agents:
            return f"Registered --agent values for {agent_library}: (none)"

        lines = [f"Registered --agent values for {agent_library}:"]
        for agent in agents:
            suffix = " (default)" if agent == f"{agent_library}_cfg_entry_point" else ""
            lines.append(f"    {agent}{suffix}")
            compatible = compatibility.get(agent)
            if compatible is not None:
                lines.append(f"      compatible presets: {', '.join(compatible)}")
        if not compatibility:
            lines.extend(["", "Preset selection does not constrain --agent for this task."])
        return "\n".join(lines)


# ============================================================================
# argv inspection (single-pass peek for tokens argparse cannot supply)
# ============================================================================


class _ArgvHelper:
    """Single-pass argv scan for tokens the parsed Namespace cannot provide.

    Needed on two counts: argparse's ``--help`` short-circuits parsing, so help
    text that depends on ``--task`` has to find it before argparse runs; and no
    argparse argument is registered for the Hydra-style selectors (see the
    module docstring), so ``presets=`` never reaches the Namespace at all.

    Attributes:
        task_name: Last ``--task`` value (matching argparse's last-wins
            semantics), or ``None`` if absent.
        help_requested: ``True`` if ``--help`` or ``-h`` is present.
        presets: Union of the names in every ``presets=NAME[,NAME,...]``
            broadcast token, empty when none is present.
        agent_explicit: ``True`` if ``--agent`` was passed in either form.
    """

    def __init__(self, argv: list[str]):
        self.task_name: str | None = None
        self.help_requested: bool = False
        self.presets: set[str] = set()
        self.agent_explicit: bool = False
        for i in range(1, len(argv)):
            token = argv[i]
            if token in ("--help", "-h"):
                self.help_requested = True
            elif token == "--task" and i + 1 < len(argv):
                self.task_name = argv[i + 1]
            elif token.startswith("--task="):
                self.task_name = token[len("--task=") :]
            elif token == "--agent" or token.startswith("--agent="):
                self.agent_explicit = True
            elif token.startswith("presets="):
                self.presets.update(name.strip() for name in token[len("presets=") :].split(",") if name.strip())

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> _ArgvHelper:
        """Scan an argv slice whose leading program name has already been stripped."""
        return cls(["", *tokens])


# ============================================================================
# Agent auto-selection (post-argparse, from task registry metadata)
# ============================================================================


def _auto_select_agent(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    task_name: str,
    agent_library: str,
    selection: _ArgvHelper,
) -> None:
    """Set ``args.agent`` when the task's metadata implies exactly one entry point.

    Tasks declare the agent configs they register, and optionally which presets
    each config is valid for (``agent_preset_compatibility``). Two rules turn
    that metadata into a selection; the second is a fallback for when the first
    does not fire:

    1. **Preset-based**: when a ``presets=`` token names a preset the task
       declares as an agent constraint, and exactly one registered entry point
       is compatible with every such preset, that entry point is used. This is
       what makes ``presets=box_discrete`` load a categorical-policy config
       instead of the Gaussian default on the Cartpole showcase tasks.

    2. **Default-absent**: when the canonical default entry point
       (``<library>_cfg_entry_point``) is not registered but exactly one other
       entry point is, that sole entry point is used. Handles tasks such as
       ``IsaacContrib-Humanoid-AMP-*`` that support only a non-default
       algorithm and so never register the PPO default.

    Does nothing when the match is absent or ambiguous, leaving the caller's own
    default resolution in charge.

    Args:
        args: Parsed namespace to update in-place.
        parser: Parser that produced *args*, used to tell a defaulted
            ``--agent`` from one the user typed.
        task_name: Gymnasium task ID used to look up the registry spec.
        agent_library: RL-library prefix (e.g. ``"skrl"``).
        selection: Scan of the argv that was actually parsed, supplying the
            ``presets=`` tokens and whether ``--agent`` was explicit.
    """
    import gymnasium as gym

    if not hasattr(args, "agent"):
        return
    # An explicit --agent always wins. Compare against the parser's default
    # rather than testing ``is None``: only skrl defaults --agent to None, while
    # rsl_rl, rl_games and sb3 default it to ``<library>_cfg_entry_point``, so an
    # ``is None`` guard would leave auto-selection unreachable for them.
    if selection.agent_explicit or args.agent != parser.get_default("agent"):
        return

    try:
        agents, compatibility = _enumerate_agents(task_name, agent_library)
    except gym.error.Error:
        # Unregistered or misspelled task ID: stay silent and let the caller's
        # own task lookup raise the real error instead of reporting it here.
        return

    # Rule 1. Only presets the task declares as agent constraints participate:
    # physics and renderer selections arrive through the same ``presets=``
    # broadcast, and including them would fail the subset test for every entry
    # point and silently leave the wrong default in place.
    declared = {preset for presets in compatibility.values() for preset in presets}
    constraining = selection.presets & declared
    if constraining:
        matches = [agent for agent, presets in compatibility.items() if constraining.issubset(set(presets))]
        if len(matches) == 1:
            args.agent = matches[0]
            return

    # Rule 2, reached whether or not a preset is active. Benchmark sweeps
    # broadcast the physics backend through the same token (``presets=newton_mjwarp``),
    # and an algorithm-only task still needs its sole entry point picked there.
    if f"{agent_library}_cfg_entry_point" not in agents and len(agents) == 1:
        args.agent = agents[0]


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


def _enumerate_agents(task_name: str, agent_library: str) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    """Return registered agents and task-declared preset compatibility."""
    import gymnasium as gym

    spec = gym.spec(task_name.split(":")[-1])
    prefix = f"{agent_library}_"
    agents = sorted(key for key in spec.kwargs if key.startswith(prefix) and key.endswith("_cfg_entry_point"))
    compatibility = spec.kwargs.get("agent_preset_compatibility", {})
    return agents, {agent: tuple(presets) for agent, presets in compatibility.items() if agent in agents}


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
