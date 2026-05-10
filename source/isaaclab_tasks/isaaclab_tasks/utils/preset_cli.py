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


def _extract_task_from_argv(argv: list[str]) -> str | None:
    """Best-effort scan of *argv* for ``--task=value`` / ``--task value``.

    Used before argparse parses, so we can pre-load the task's env config
    (which transitively imports its backends and populates the registry)
    in time for ``--help`` enrichment and validation. Mirrors argparse's
    semantics in the cases that matter for the pre-load:

    * Returns the LAST occurrence to match argparse's last-wins behavior
      for repeated single-value flags. This matters for ``--help``, which
      exits before the post-parse reload path runs.
    * Stops scanning at the ``--`` end-of-options marker so a task name
      after ``--`` (which argparse leaves as a positional) doesn't pre-empt
      one before it.

    Limitation: argparse's default ``allow_abbrev=True`` accepts unambiguous
    prefixes (e.g. ``--tas Foo``), but the pre-scan only recognizes the
    literal ``--task`` / ``--task=``. The non-help path covers this
    automatically -- the post-parse reload reads ``args.task`` and reloads
    when it differs from ``pre_task`` -- but ``--help`` exits before that
    runs, so ``train.py --tas Foo --help`` shows generic help (no
    task-specific variants). Use the full ``--task`` in ``--help``
    invocations, or pass ``allow_abbrev=False`` to your parser.
    """
    last: str | None = None
    for i, token in enumerate(argv):
        if token == "--":
            break
        if token == "--task" and i + 1 < len(argv):
            last = argv[i + 1]
        elif token.startswith("--task="):
            last = token[len("--task=") :]
    return last


def _load_task_env_cfg(task_name: str) -> object | None:
    """Load *task_name*'s env config; return an instance or ``None``.

    Side effect: importing the env config module triggers the backend cfg
    imports referenced by the task, which fires the :func:`register`
    decorators that populate :class:`PresetRegistry`.

    On import failure the error is written to ``stderr`` and ``None`` is
    returned, so ``--help`` and validation can still proceed with whatever
    is already registered. This is loud-by-default: silent swallow would
    make a typo like ``--task IsacaCartpole`` look like the registry is
    empty for that task, which is misleading.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    try:
        env_cfg = load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")
    except Exception as exc:  # noqa: BLE001 -- broad on purpose: gym, carb, importlib all raise differently
        sys.stderr.write(
            f"warning: could not load task {task_name!r} for preset validation: {type(exc).__name__}: {exc}\n"
        )
        return None
    if isinstance(env_cfg, type):
        try:
            env_cfg = env_cfg()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"warning: could not instantiate {task_name!r} env config: {type(exc).__name__}: {exc}\n")
            return None
    return env_cfg


def _canonical_and_target(value: object) -> tuple[str | None, PresetTarget | None]:
    """Look up the canonical preset name + target for *value*'s class.

    Walks the MRO for ``_preset_name`` / ``_preset_target`` (stamped by
    :func:`register`). Falls back to ``value.solver_cfg`` so wrappers like
    ``NewtonCfg`` (which holds a registered solver-cfg) still resolve.
    Returns ``(None, None)`` when nothing in the chain is decorated.
    """
    for klass in type(value).__mro__:
        if "_preset_name" in klass.__dict__:
            return klass.__dict__["_preset_name"], klass.__dict__["_preset_target"]
    inner = getattr(value, "solver_cfg", None)
    if inner is not None:
        for klass in type(inner).__mro__:
            if "_preset_name" in klass.__dict__:
                return klass.__dict__["_preset_name"], klass.__dict__["_preset_target"]
    return None, None


def _preset_alternatives_view(node: object) -> dict[str, object]:
    """Return ``{name: value}`` for every alternative on a :class:`PresetCfg`.

    Mirrors :func:`isaaclab_tasks.utils.hydra._preset_fields` exactly:

    * Class-level values take priority over instance values for declared
      ``__dataclass_fields__`` -- robot-specific cfg modules reassign field
      values on the class after instances are already constructed, and the
      Hydra resolver applies those class-level values when picking an
      alternative. The variant walker must agree.
    * Picks up class-only attributes (added outside the dataclass mechanism)
      when they aren't dunder, callable, or already covered.

    Used ONLY when reading alternatives off a single ``PresetCfg`` node;
    tree-level recursion uses :func:`_walk_cfg_items` instead, so we stay
    aligned with whichever traversal Hydra uses for each phase.
    """
    cls = type(node)
    out: dict[str, object] = {}
    fields = getattr(node, "__dataclass_fields__", None)
    if fields is not None:
        for fname in fields:
            cls_val = getattr(cls, fname, None)
            out[fname] = cls_val if cls_val is not None else getattr(node, fname, None)
    for attr in vars(cls):
        if attr.startswith("_") or attr in out or callable(getattr(cls, attr, None)):
            continue
        out[attr] = getattr(cls, attr)
    return out


def _walk_cfg_items(node: object):
    """Yield ``(name, value)`` pairs for tree recursion, mirroring
    :func:`isaaclab_tasks.utils.hydra._walk_cfg`.

    For dicts: items as-is. For other objects: every non-underscore
    attribute reachable via ``getattr`` -- which is instance-first with
    class fallback. This deliberately differs from
    :func:`_preset_alternatives_view` (class-first) because it matches
    how :func:`resolve_presets` finds nested ``PresetCfg`` nodes. If we
    advertised a class-level override that the resolver would never see,
    the typed-flag layer would accept names ``resolve_presets`` then
    can't apply.
    """
    if isinstance(node, dict):
        return list(node.items())
    items: list[tuple[str, object]] = []
    for n in dir(node):
        if n.startswith("_"):
            continue
        val = getattr(node, n, None)
        if val is None:
            continue
        items.append((n, val))
    return items


def _collect_task_variants(env_cfg: object) -> dict[PresetTarget, set[str]]:
    """Walk *env_cfg* and harvest variant field names from every :class:`PresetCfg`.

    Returns ``{target: set[name]}``. Field ``"default"`` is skipped because
    it holds the active selection rather than an alternative.

    Strict-anchor rule: within a single ``PresetCfg``, fields are grouped
    by the canonical name of their value's class. A group is accepted only
    when at least one of its field names *is* the canonical name; that
    field anchors the group and other fields in the group are accepted
    as variants. A group with no canonical anchor is dropped from the
    variants set, so e.g.::

        @configclass
        class PhysicsCfg(PresetCfg):
            default: ... = MISSING
            newton_mjwarp2: MjwarpCfg = MjwarpCfg(...)  # variant only, no anchor

    does *not* make ``--physics newton_mjwarp2`` selectable. The user
    must add a sibling ``newton_mjwarp: MjwarpCfg = ...`` field to anchor
    the group; then ``newton_mjwarp2`` is accepted alongside it. This
    matches the cross-env drift lint, so the CLI surface and the lint
    agree on what counts as a valid variant.
    """
    from isaaclab_tasks.utils.hydra import PresetCfg

    variants: dict[PresetTarget, set[str]] = {}

    def _visit(node: object) -> None:
        if isinstance(node, PresetCfg):
            # (target, canonical) -> [fname, ...]. Keying by target as well
            # as canonical keeps groups cleanly separated even if a name
            # were ever reused across targets. Use the alternatives view
            # (class-first) here -- matches hydra._preset_fields, which
            # is what the resolver uses to read PresetCfg fields.
            alternatives = _preset_alternatives_view(node)
            by_canonical: dict[tuple[PresetTarget, str], list[str]] = {}
            for fname, value in alternatives.items():
                if fname == "default" or value is None:
                    continue
                canonical, target = _canonical_and_target(value)
                if canonical is None or target is None:
                    continue
                by_canonical.setdefault((target, canonical), []).append(fname)

            for (target, canonical), fnames in by_canonical.items():
                if canonical not in fnames:
                    # No anchor: drop the whole group. The cross-env drift
                    # lint flags this; the CLI must not legitimize it by
                    # accepting the variant names.
                    continue
                variants.setdefault(target, set()).update(fnames)

            # Recurse into the alternatives we just read (class-first
            # values), not via _walk_cfg_items (instance-first) -- the
            # resolver picks one of these class-first alternatives and
            # only AFTER picking does it descend via _walk_cfg into the
            # picked one. A class-level override of an alternative whose
            # value contains nested PresetCfgs would otherwise be missed.
            for value in alternatives.values():
                if value is None:
                    continue
                if isinstance(value, PresetCfg) or hasattr(value, "__dataclass_fields__") or isinstance(value, dict):
                    _visit(value)
            return

        # Non-PresetCfg recursion: instance-first via _walk_cfg_items.
        # Mirrors hydra._walk_cfg which uses getattr against the cfg
        # instance, so nothing the resolver wouldn't reach is advertised.
        for _key, val in _walk_cfg_items(node):
            if isinstance(val, PresetCfg) or hasattr(val, "__dataclass_fields__") or isinstance(val, dict):
                _visit(val)

    _visit(env_cfg)
    return variants


def _validate_typed_flag(target: PresetTarget, value: str | None, variants: set[str]) -> str | None:
    """Reject unknown names; normalize legacy aliases.

    A name is valid when it is in :meth:`PresetRegistry.names_for` for
    *target* (a registered backend) or in *variants* (a field name in the
    selected task's :class:`PresetCfg` for *target*). A task-local variant
    that happens to share a deprecated alias name (e.g. a real ``newton``
    field on the task's ``PhysicsCfg``) is preserved as-is and *not*
    rewritten to the alias's canonical -- the variant shadows the alias.

    Returns the canonical name (possibly normalized from a legacy alias)
    or ``None`` when *value* is ``None``. Raises ``SystemExit`` with a
    helpful message when the name is not valid.
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


def _help_text(target: PresetTarget, valid: set[str], task: str | None) -> str:
    """Argparse ``help=`` string showing valid preset names for *target*."""
    capitalized = target.value.capitalize()
    if not valid:
        return f"{capitalized} preset name. Pass '--task=<X> --help' to list valid names for task X."
    listing = ", ".join(sorted(valid))
    scope = f"for task {task!r}" if task else "registered"
    return f"{capitalized} preset name. Available ({scope}): {listing}."


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
    5. Validate each typed flag against ``names_for(target) | variants[target]``.
    6. Collect names + merge with any pre-existing ``presets=...`` token
       from *remaining*; dedupe; rewrite ``sys.argv`` so the Hydra layer
       sees one ``presets=<csv>`` token followed by leftover Hydra args.

    Returns the parsed argparse namespace; ``sys.argv`` is mutated in place.
    """
    # Lazy: AppLauncher pulls in Isaac Sim. Keep the import inside this
    # function so this module is importable without Sim being available.
    from isaaclab.app import AppLauncher

    # Pre-scan: load the task's env config now so the registry + variant
    # set are ready when we build help strings and validate.
    pre_task = _extract_task_from_argv(sys.argv[1:])
    env_cfg = _load_task_env_cfg(pre_task) if pre_task else None
    loaded_task = pre_task if env_cfg is not None else None
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

    # ``args.task`` may not exist when --task is in a subparser or wasn't
    # added at all -- fall back to the pre-scan value so we still validate.
    task_name = getattr(args, "task", None) or pre_task

    if any_typed and not task_name:
        raise SystemExit("error: --physics/--renderer require --task=<task-name> to validate against.")

    # Reload when the pre-scan didn't yield an env cfg (unusual --task form
    # or load failure) OR the parsed task name differs from what we loaded
    # (subparser layouts, repeated --task flags). Always reset variants from
    # the new env_cfg so a failed reload doesn't leave stale variants from
    # the previous task. A successful load with no variants for the SAME
    # task is a legitimate empty result and doesn't retrigger.
    if any_typed and task_name and (env_cfg is None or task_name != loaded_task):
        env_cfg = _load_task_env_cfg(task_name)
        task_variants = _collect_task_variants(env_cfg) if env_cfg is not None else {}
        loaded_task = task_name if env_cfg is not None else None

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
