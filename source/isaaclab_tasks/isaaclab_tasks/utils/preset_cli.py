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
    in time for ``--help`` enrichment and validation.
    """
    for i, token in enumerate(argv):
        if token == "--task" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--task="):
            return token[len("--task=") :]
    return None


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


def _collect_task_variants(env_cfg: object) -> dict[PresetTarget, set[str]]:
    """Walk *env_cfg* and harvest field names from every :class:`PresetCfg`.

    Returns ``{target: set[name]}`` where *target* comes from each field
    value's ``_preset_target`` (set by :func:`register`) and *name* is the
    field name in the parent ``PresetCfg``. Field ``"default"`` is skipped
    because it holds the active selection rather than an alternative.

    These names are the ones the user wrote on their ``PresetCfg``, so they
    are always valid CLI choices for that task even when not in the global
    registry. ``setup_cli`` unions them with :meth:`PresetRegistry.names_for`
    when validating typed flags and rendering ``--help``.
    """
    from isaaclab_tasks.utils.hydra import PresetCfg

    variants: dict[PresetTarget, set[str]] = {}

    def _visit(node: object) -> None:
        if isinstance(node, PresetCfg):
            for fname in node.__dataclass_fields__:
                if fname == "default":
                    continue
                value = getattr(node, fname, None)
                if value is None:
                    continue
                target = None
                for klass in type(value).__mro__:
                    if "_preset_target" in klass.__dict__:
                        target = klass.__dict__["_preset_target"]
                        break
                if target is not None:
                    variants.setdefault(target, set()).add(fname)

        # Recurse: dataclasses, dicts, and PresetCfg children.
        items: list[tuple[str, object]] = []
        if isinstance(node, dict):
            items = list(node.items())
        elif hasattr(node, "__dataclass_fields__"):
            for name in node.__dataclass_fields__:
                items.append((name, getattr(node, name, None)))
        for _key, val in items:
            if val is None:
                continue
            if hasattr(val, "__dataclass_fields__") or isinstance(val, (dict, PresetCfg)):
                _visit(val)

    _visit(env_cfg)
    return variants


def _validate_typed_flag(target: PresetTarget, value: str | None, variants: set[str]) -> str | None:
    """Reject unknown names; normalize legacy aliases.

    A name is valid when it is in :meth:`PresetRegistry.names_for` for
    *target* (a registered backend) or in *variants* (a field name in the
    selected task's :class:`PresetCfg` for *target*).

    Returns the canonical name (possibly normalized from a legacy alias)
    or ``None`` when *value* is ``None``. Raises ``SystemExit`` with a
    helpful message when the name is not valid.
    """
    if value is None:
        return None
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

    if any_typed and not args.task:
        raise SystemExit("error: --physics/--renderer require --task=<task-name> to validate against.")

    # Defensive: if the pre-scan didn't catch --task (e.g., it appeared in
    # an unusual form), or env-cfg load failed and the user fixed it via
    # some other mechanism, retry now using the parsed args.task.
    if any_typed and not task_variants and args.task:
        env_cfg = _load_task_env_cfg(args.task)
        if env_cfg is not None:
            task_variants = _collect_task_variants(env_cfg)

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
