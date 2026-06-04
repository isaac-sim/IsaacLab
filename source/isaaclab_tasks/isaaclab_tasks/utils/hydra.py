# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hydra utilities with REPLACE-only preset system.

This module bypasses Hydra's default MERGE behavior for config groups.
Instead, when a preset is selected, the entire config section is REPLACED
with the preset -- no field merging.

Presets are declared by subclassing :class:`PresetCfg` (or using the
:func:`preset` factory for scalars). The system recursively discovers all
presets and their paths automatically, including inside dict-valued fields.

Override categories (applied in order):
    1. Global presets: ``presets=inference,newton_mjwarp`` -- apply everywhere matching
    2. Typed selectors: ``physics=newton_mjwarp`` / ``renderer=NAME`` -- like a
       global preset, but must resolve against a config of that type or it errors
    3. Path presets: ``env.backend=newton_mjwarp`` -- REPLACE specific section
    4. Preset-path scalars: ``env.backend.dt=0.001`` -- handled by us
    5. Global scalars: ``env.decimation=10`` -- handled by Hydra

Example usage::

    presets=newton_mjwarp env.backend.dt=0.001 env.decimation=10
"""

import ast
import functools
import sys
import warnings
from collections import deque
from collections.abc import Callable, Mapping

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices
from isaaclab.utils.configclass import configclass

from .preset_target import PresetTarget

_LITERAL_MAP = {"true": True, "false": False, "none": None, "null": None}


def _user_stacklevel() -> int:
    """Compute a ``warnings.warn`` stacklevel that lands on the first frame
    outside the ``isaaclab_tasks.utils`` package, so deprecation messages
    cite user code rather than internal utility frames.

    Walks at most a small bounded number of frames; if no out-of-package
    frame is found within the bound (frozen modules, exec'd contexts, or
    oddly named ``__name__`` globals), falls back to ``stacklevel=2`` so
    the warning at least jumps out of the helper that called it.

    Package-scoped (not file-scoped) so callers in any module under
    ``isaaclab_tasks.utils.*`` (``hydra``, ``parse_cfg``, ...) get the same
    "skip our own internals" behavior without duplicating the walk.
    """
    max_walk = 16
    level = 1
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__", "").startswith(__package__):
        level += 1
        frame = frame.f_back
        if level > max_walk:
            return 2
    return level


def _known_preset_names(presets: dict) -> set[str]:
    """Return all preset names declared in a collected preset dictionary."""
    return {name for section in presets.values() for fields in section.values() for name in fields}


def _normalize_preset_name(name: str, known_names: set[str]) -> str:
    """Map a deprecated preset name to its replacement and emit a warning.

    Returns ``name`` unchanged when:
        * ``name`` is not a deprecated alias, or
        * the replacement is not declared in ``known_names`` (so the user-supplied
          value can flow into the standard "unknown preset" error path, where
          :func:`_format_unknown_presets_error` will surface the rename), or
        * ``name`` is itself a real field in ``known_names`` (a user-defined preset
          legitimately reusing the deprecated spelling shadows the alias).
    """
    replacement = PresetTarget.all_legacy_aliases().get(name)
    if replacement is None or replacement not in known_names or name in known_names:
        return name
    warnings.warn(
        f"Preset '{name}' is deprecated. Use '{replacement}' instead.",
        FutureWarning,
        stacklevel=_user_stacklevel(),
    )
    return replacement


@configclass
class PresetCfg:
    """Base class for declarative preset definitions.

    Subclass this and define fields as preset options.
    The field named ``default`` holds the config instance used
    when no CLI override is given. All other fields are named
    alternative presets.

    Example::

        @configclass
        class PhysicsCfg(PresetCfg):
            default: PhysxCfg = PhysxCfg()
            newton_mjwarp: NewtonCfg = NewtonCfg()

    The preset *name* (``newton_mjwarp``) is decoupled from the config class
    (``NewtonCfg``): the class describes the Newton backend, while the field
    name labels which solver variant this entry selects.

    **Class-local helpers (underscore convention).** Names prefixed with
    ``_`` and callables (nested classes, methods) are skipped by the
    resolver and are NOT registered as variants. Use this to keep shared
    helpers adjacent to the variants that need them, without polluting the
    module namespace::

        @configclass
        class MultiBackendCameraCfg(PresetCfg):
            # Class-local helper -- not a variant.
            _ROTATED_OFFSET = CameraCfg.OffsetCfg(rot=(1, 0, 0, 0), ...)

            rgb = CameraCfg(data_types=["rgb"])
            albedo = CameraCfg(data_types=["albedo"], offset=_ROTATED_OFFSET)
            default = rgb
    """

    def __getattr__(self, name: str):
        """Alias a deprecated preset name to its replacement field.

        Raises ``AttributeError`` for any other missing attribute so that
        ``hasattr`` and standard introspection keep working unchanged. The
        replacement is only returned when the deprecated name is *not* itself a
        real field on the subclass, so a user redefining the deprecated name
        shadows the alias.
        """
        replacement = PresetTarget.all_legacy_aliases().get(name)
        fields = getattr(type(self), "__dataclass_fields__", {})
        if replacement is not None and replacement in fields and name not in fields:
            warnings.warn(
                f"Preset '{name}' is deprecated. Use '{replacement}' instead.",
                FutureWarning,
                stacklevel=_user_stacklevel(),
            )
            return getattr(self, replacement)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")


def preset(**options) -> PresetCfg:
    """Create a :class:`PresetCfg` instance from keyword arguments.

    A convenience factory that dynamically builds a ``PresetCfg`` subclass
    with one field per keyword argument, then returns an instance of it.
    The caller **must** supply a ``default`` key.

    Example::

        armature = preset(default=0.0, newton_mjwarp=0.01)
        # Equivalent to:
        # @configclass
        # class _Preset(PresetCfg):
        #     default: float = 0.0
        #     newton_mjwarp: float = 0.01
        # armature = _Preset()

    Args:
        **options: Preset alternatives keyed by name.  Must include ``default``.

    Returns:
        A ``PresetCfg`` instance whose fields are the supplied options.

    Raises:
        ValueError: If ``default`` is not provided.
    """
    if "default" not in options:
        raise ValueError("preset() requires a 'default' keyword argument.")
    annotations = {k: type(v) if v is not None else object for k, v in options.items()}
    ns = {"__annotations__": annotations, **options}
    cls = configclass(type("_Preset", (PresetCfg,), ns))
    return cls()


def _preset_fields(preset_obj) -> dict:
    """Extract all alternatives from a :class:`PresetCfg`, class attrs over instance.

    Class-level values take priority because robot-specific modules
    (e.g. ``joint_pos_env_cfg.py``) reassign fields on the class after
    instances are already created.
    """
    cls = type(preset_obj)
    d = {}
    for fn in preset_obj.__dataclass_fields__:
        cls_val = getattr(cls, fn, None)
        d[fn] = cls_val if cls_val is not None else getattr(preset_obj, fn)
    for attr in vars(cls):
        if attr.startswith("_") or attr in d or callable(getattr(cls, attr)):
            continue
        d[attr] = getattr(cls, attr)
    return d


def _iter_cfg_items(cfg):
    if isinstance(cfg, Mapping):
        return cfg.items()
    if isinstance(cfg, list):
        return enumerate(cfg)
    return ((n, v) for n in dir(cfg) if not n.startswith("_") for v in [getattr(cfg, n, None)] if v is not None)


def _is_walkable_cfg(cfg) -> bool:
    return hasattr(cfg, "__dataclass_fields__") or isinstance(cfg, (Mapping, list))


def _walk_cfg(cfg, path: str, on_preset: Callable) -> None:
    """Depth-first walk of a config tree, calling *on_preset(parent, key, obj, path)*
    for every :class:`PresetCfg` node.  Recurses through dataclass attrs, dicts,
    nested dicts, and lists transparently."""
    for key, val in _iter_cfg_items(cfg):
        child_path = f"{path}.{key}" if path else str(key)
        if isinstance(val, PresetCfg):
            on_preset(cfg, key, val, child_path)
        elif _is_walkable_cfg(val):
            _walk_cfg(val, child_path, on_preset)


def collect_presets(cfg, path: str = "") -> dict:
    """Recursively discover :class:`PresetCfg` nodes in the config tree.

    Walks dataclass fields and dict values at any nesting depth.

    Args:
        cfg: A configclass instance to walk.
        path: Current path prefix (used during recursion).

    Returns:
        Dict mapping dotted paths to preset dicts, e.g.:
        ``{"backend": {"default": PhysxCfg(), "newton_mjwarp": NewtonCfg()}}``
    """
    result = {}

    def _record(preset_obj, preset_path):
        fields = _preset_fields(preset_obj)
        result[preset_path] = fields
        for alt in fields.values():
            if hasattr(alt, "__dataclass_fields__"):
                result.update(collect_presets(alt, preset_path))
            elif isinstance(alt, dict):
                for v in alt.values():
                    if _is_walkable_cfg(v):
                        result.update(collect_presets(v, preset_path))
            elif isinstance(alt, list):
                for v in alt:
                    if _is_walkable_cfg(v):
                        result.update(collect_presets(v, preset_path))

    if isinstance(cfg, PresetCfg):
        _record(cfg, path)
        return result

    _walk_cfg(cfg, path, lambda _p, _k, obj, cp: _record(obj, cp))
    return result


# ============================================================================
# Preset resolution
# ============================================================================


def _pick_alternative(
    preset_obj: PresetCfg,
    selected,
    path: str = "",
    explicit_name: str | None = None,
    consumed_selected: set[str] | None = None,
    typed_hits: dict[str, set[PresetTarget]] | None = None,
):
    """Choose the best alternative from a PresetCfg.

    Priority: first match in ``selected``, then ``default`` (preferring
    class-level over instance-level).

    Raises:
        ValueError: If no matching name and no ``default`` field exists.
    """
    fields = _preset_fields(preset_obj)
    field_names = set(fields)
    if explicit_name is not None:
        explicit_name = _normalize_preset_name(explicit_name, field_names)
        if explicit_name in fields:
            return fields[explicit_name]
        avail = list(fields)
        hint = ""
        if explicit_name in PresetTarget.all_legacy_aliases():
            replacement = PresetTarget.all_legacy_aliases()[explicit_name]
            hint = (
                f" '{explicit_name}' was renamed to '{replacement}'; this path does not declare '{replacement}' either."
            )
        raise ValueError(f"Unknown preset '{explicit_name}' for {path}. Available: {avail}.{hint}")

    match_name = None
    match_value = None
    for name in selected:
        raw_name = name
        name = _normalize_preset_name(raw_name, field_names)
        if name not in fields or name == match_name:
            continue
        val = fields[name]
        if consumed_selected is not None:
            consumed_selected.add(raw_name)
            consumed_selected.add(name)
        if typed_hits is not None:
            # record which typed targets (physics/renderer) this name landed on
            targets = {t for t in PresetTarget if t.base_classes and isinstance(val, t.base_classes)}
            if targets:
                typed_hits.setdefault(raw_name, set()).update(targets)
                typed_hits.setdefault(name, set()).update(targets)
        if match_name is not None:
            if match_value is not val and match_value != val:
                raise ValueError(
                    f"Conflicting global presets: '{match_name}' and '{name}' both define preset for '{path}'"
                )
        match_name, match_value = name, val
    if match_name is not None:
        return match_value
    if "default" in fields:
        return fields["default"]
    raise ValueError(
        f"PresetCfg {type(preset_obj).__name__} at '{path}' has no 'default' field "
        f"and none of the selected presets {selected} match its fields {set(fields.keys())}."
    )


def _resolve_active_presets(
    cfg,
    selected=(),
    explicit: dict[str, str] | None = None,
    root_path: str = "",
    *,
    strict_explicit: bool = True,
    consumed_selected: set[str] | None = None,
    typed_hits: dict[str, set[PresetTarget]] | None = None,
    consumed_explicit: set[str] | None = None,
):
    """Resolve presets by walking only the currently active tree.

    Preset alternatives are choice nodes. Once a choice is resolved, only the
    selected replacement is queued for further traversal, so inactive sibling
    branches cannot contribute descendant presets.
    """
    explicit = explicit or {}
    consumed_explicit = consumed_explicit if consumed_explicit is not None else set()

    def resolve_chain(preset_obj: PresetCfg, path: str):
        seen: set[int] = set()
        val = preset_obj
        while isinstance(val, PresetCfg):
            if id(val) in seen:
                raise ValueError(
                    f"Cyclic PresetCfg chain detected at '{path}': {type(val).__name__} was already visited."
                )
            seen.add(id(val))
            val = _pick_alternative(
                val,
                selected,
                path=path,
                explicit_name=explicit.get(path),
                consumed_selected=consumed_selected,
                typed_hits=typed_hits,
            )
        return val

    if isinstance(cfg, PresetCfg):
        if root_path in explicit:
            consumed_explicit.add(root_path)
        cfg = resolve_chain(cfg, root_path or "<root>")

    queue = deque([(root_path, cfg)])
    while queue:
        path, obj = queue.popleft()
        if not _is_walkable_cfg(obj):
            continue
        for key, val in _iter_cfg_items(obj):
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(val, PresetCfg):
                if child_path in explicit:
                    consumed_explicit.add(child_path)
                resolved = resolve_chain(val, child_path or "<root>")
                if isinstance(obj, list):
                    obj[int(key)] = resolved
                elif isinstance(obj, dict):
                    obj[key] = resolved
                else:
                    setattr(obj, key, resolved)
                if _is_walkable_cfg(resolved):
                    queue.append((child_path, resolved))
            elif _is_walkable_cfg(val):
                queue.append((child_path, val))

    missing = sorted(set(explicit) - consumed_explicit)
    if strict_explicit and missing:
        raise ValueError(f"Unknown or inactive preset group(s): {', '.join(missing)}")
    return cfg


def resolve_presets(cfg, selected=()):
    """Replace every :class:`PresetCfg` in the tree with the best alternative.

    For each ``PresetCfg`` found during an active-tree breadth-first walk:

    1. Pick the first name from *selected* that exists as a field on the
       preset, otherwise fall back to ``default``.
    2. Replace the preset in its parent (dict key or dataclass attr).
    3. Continue walking the replacement (which may contain more presets).

    Args:
        cfg: A configclass, dict, or PresetCfg to resolve in-place.
        selected: Set of preset names chosen by the user (e.g. from CLI
            ``presets=peg_insert_4mm,eval``).

    Returns:
        The resolved ``cfg`` (possibly a different object if the root itself
        was a PresetCfg).
    """
    return _resolve_active_presets(cfg, selected)


# ============================================================================
# CLI / Hydra integration
# ============================================================================


def _run_hydra(task, env_cfg, agent_cfg, hydra_args, callback):
    """Shared Hydra entry point for :func:`resolve_task_config` and :func:`hydra_task_config`."""
    if not hydra_args:
        env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
        callback(env_cfg, agent_cfg)
        return

    original_argv, sys.argv = sys.argv, [sys.argv[0]] + hydra_args

    @hydra.main(config_path=None, config_name=task, version_base="1.3")
    def hydra_main(hydra_cfg, env_cfg=env_cfg, agent_cfg=agent_cfg):
        hydra_cfg = replace_strings_with_slices(OmegaConf.to_container(hydra_cfg, resolve=True))
        env_cfg.from_dict(hydra_cfg["env"])
        env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
        if isinstance(agent_cfg, dict) or agent_cfg is None:
            agent_cfg = hydra_cfg["agent"]
        else:
            agent_cfg.from_dict(hydra_cfg["agent"])
        callback(env_cfg, agent_cfg)

    try:
        hydra_main()
    finally:
        sys.argv = original_argv


def resolve_task_config(task_name: str, agent_cfg_entry_point: str):
    """Resolve env and agent configs with Hydra overrides, presets, and scalars fully applied.

    Safe to call before Kit is launched -- callable config values are stored as
    :class:`~isaaclab.utils.string.ResolvableString` and resolved lazily on
    first use, so no implementation modules are imported eagerly.

    Args:
        task_name: Task name (e.g., "Isaac-Velocity-Flat-Anymal-C-v0").
        agent_cfg_entry_point: Agent config entry point key (e.g., "rsl_rl_cfg_entry_point").

    Returns:
        Tuple of (env_cfg, agent_cfg) fully resolved.
    """
    task = task_name.split(":")[-1]
    env_cfg, agent_cfg, hydra_args = register_task(task, agent_cfg_entry_point)
    resolved = {}
    _run_hydra(task, env_cfg, agent_cfg, hydra_args, lambda e, a: resolved.update(env_cfg=e, agent_cfg=a))
    return resolved["env_cfg"], resolved["agent_cfg"]


def hydra_task_config(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Decorator for Hydra config with REPLACE-only preset semantics.

    Args:
        task_name: Task name (e.g., "Isaac-Reach-Franka-v0")
        agent_cfg_entry_point: Agent config entry point key

    Returns:
        Decorated function receiving ``(env_cfg, agent_cfg, *args, **kwargs)``
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            task = task_name.split(":")[-1]
            env_cfg, agent_cfg, hydra_args = register_task(task, agent_cfg_entry_point)
            _run_hydra(task, env_cfg, agent_cfg, hydra_args, lambda e, a: func(e, a, *args, **kwargs))

        return wrapper

    return decorator


def _format_unknown_presets_error(unknown: set[str], name_to_paths: dict[str, list[str]], max_paths: int = 5) -> str:
    """Build a readable error message grouping presets by identical path fingerprints.

    When an unknown name matches a deprecated alias (e.g. ``newton``), the
    message explicitly calls out the rename so users updating from older
    tutorials or scripts get an actionable hint instead of a bare "unknown".
    """
    fingerprint_to_names: dict[tuple[str, ...], list[str]] = {}
    for name, paths in name_to_paths.items():
        key = tuple(sorted(paths))
        fingerprint_to_names.setdefault(key, []).append(name)

    lines = [f"Unknown preset(s): {', '.join(sorted(unknown))}"]
    deprecated_hits = sorted(name for name in unknown if name in PresetTarget.all_legacy_aliases())
    for legacy in deprecated_hits:
        replacement = PresetTarget.all_legacy_aliases()[legacy]
        lines.append(f"  '{legacy}' was renamed to '{replacement}'; this task does not declare '{replacement}' either.")
    lines += [
        "",
        "Available presets (grouped by affected paths):",
        "",
    ]
    for paths_tuple in sorted(fingerprint_to_names, key=lambda k: fingerprint_to_names[k][0]):
        names = sorted(fingerprint_to_names[paths_tuple])
        if len(names) <= 30:
            lines.append(f"  {', '.join(names)}")
        else:
            lines.append(f"  {', '.join(names[:25])}, ... ({len(names)} total)")
        shown = list(paths_tuple[:max_paths])
        for p in shown:
            lines.append(f"    -> {p}")
        remaining = len(paths_tuple) - len(shown)
        if remaining > 0:
            lines.append(f"    ... ({remaining} more)")
        lines.append("")
    return "\n".join(lines)


def _validate_typed_presets(
    requested: dict[PresetTarget, set[str]],
    typed_hits: dict[str, set[PresetTarget]],
) -> None:
    """Check that each typed selector landed on a config of its own type.

    A typed selector (``physics=NAME`` / ``renderer=NAME``) is an explicit
    request for a backend of that type, so ``NAME`` must replace at least one
    config of that type during resolution. If it only matched unrelated presets
    that happen to share the name (a scalar, a sensor variant), the backend
    silently stays unchanged, so raise. The free-form ``presets=NAME`` broadcast
    is intentionally *not* checked -- there the user makes no typing claim.

    Raises:
        ValueError: If a ``physics=`` / ``renderer=`` name never resolved
            against a config of that target's type.
    """
    aliases = PresetTarget.all_legacy_aliases()
    missing = sorted(
        (t.value, n) for t, ns in requested.items() for n in ns if t not in typed_hits.get(aliases.get(n, n), set())
    )
    if missing:
        clauses = ", ".join(f"{label}={name}" for label, name in missing)
        raise ValueError(
            f"Typed preset selector(s) {clauses} did not match any preset of that type for this task. "
            "The name only matched unrelated presets (or nothing), so the backend would stay unchanged. "
            "Use a task that declares it on the matching config, or drop the selector."
        )


def register_task(task_name: str, agent_entry: str) -> tuple:
    """Load configs, collect presets recursively, register base config to Hydra.

    Presets are collected from nested configclasses and stored separately -
    NOT registered as Hydra groups to avoid Hydra's merge behavior.

    Returns:
        Tuple of ``(env_cfg, agent_cfg, hydra_args)`` where presets have been
        resolved and ``hydra_args`` contains the remaining non-preset Hydra
        overrides.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_name, agent_entry) if agent_entry else None

    # CLI preset tokens: ``presets=NAME[,...]`` broadcasts (no typing claim),
    # while ``physics=NAME`` / ``renderer=NAME`` are typed selectors that must
    # resolve against a config of that type (enforced after resolution).
    typed_labels = {target.value: target for target in PresetTarget if target.base_classes}
    global_presets: list[str] = []
    requested_targets: dict[PresetTarget, set[str]] = {}
    override_items: list[tuple[str, str, str]] = []
    hydra_args: list[str] = []
    for arg in sys.argv[1:]:
        if "=" not in arg:
            hydra_args.append(arg)
            continue
        key, val = arg.split("=", 1)
        token = key.lstrip("-")
        if token == PresetTarget.DOMAIN.value:
            global_presets.extend(v.strip() for v in val.split(",") if v.strip())
        elif token in typed_labels:
            for name in (v.strip() for v in val.split(",") if v.strip()):
                global_presets.append(name)
                requested_targets.setdefault(typed_labels[token], set()).add(name)
        else:
            override_items.append((key, val, arg))

    explicit = {key: val for key, val, _arg in override_items}
    consumed_presets: set[str] = set()
    typed_hits: dict[str, set[PresetTarget]] = {}
    consumed_explicit: set[str] = set()
    env_explicit = {path: name for path, name in explicit.items() if path == "env" or path.startswith("env.")}
    agent_explicit = {path: name for path, name in explicit.items() if path == "agent" or path.startswith("agent.")}
    env_cfg = _resolve_active_presets(
        env_cfg,
        global_presets,
        env_explicit,
        root_path="env",
        strict_explicit=False,
        consumed_selected=consumed_presets,
        typed_hits=typed_hits,
        consumed_explicit=consumed_explicit,
    )
    if agent_cfg is not None:
        agent_cfg = _resolve_active_presets(
            agent_cfg,
            global_presets,
            agent_explicit,
            root_path="agent",
            strict_explicit=False,
            consumed_selected=consumed_presets,
            typed_hits=typed_hits,
            consumed_explicit=consumed_explicit,
        )

    unknown_presets = set(global_presets) - consumed_presets
    if unknown_presets:
        # Build the full discovery table only on the error path, or when a
        # selected name applies only to inactive branches and therefore has no
        # effect in the active-tree walk.
        all_presets = {
            "env": collect_presets(load_cfg_from_registry(task_name, "env_cfg_entry_point")),
            "agent": collect_presets(load_cfg_from_registry(task_name, agent_entry)) if agent_entry else {},
        }
        name_to_paths: dict[str, list[str]] = {}
        for sec, sec_presets in all_presets.items():
            for path, fields in sec_presets.items():
                full = f"{sec}.{path}" if path else sec
                for name in fields:
                    name_to_paths.setdefault(name, []).append(full)
        known_names = set(name_to_paths)
        unknown = {_normalize_preset_name(name, known_names) for name in unknown_presets} - known_names
        if unknown:
            display = {n: p for n, p in name_to_paths.items() if n != "default"}
            raise ValueError(_format_unknown_presets_error(unknown, display))

    # Typed selectors (physics=/renderer=) must have landed on a cfg of their type
    _validate_typed_presets(requested_targets, typed_hits)

    cfgs = {"env": env_cfg, "agent": agent_cfg}
    for key, val, arg in override_items:
        if key in consumed_explicit:
            continue
        if key.startswith(("env.", "agent.")) and not key.endswith("+"):
            sec, path = key.split(".", 1)
            _setattr(cfgs[sec], path, _parse_val(val))
        else:
            hydra_args.append(arg)

    if not hydra_args:
        return env_cfg, agent_cfg, hydra_args

    # Convert to dict for Hydra (handle gym spaces and slices)
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    agent_dict = agent_cfg.to_dict() if agent_cfg is not None and hasattr(agent_cfg, "to_dict") else agent_cfg
    env_dict = env_cfg.to_dict()  # type: ignore[union-attr]
    cfg_dict = replace_slices_with_strings({"env": env_dict, "agent": agent_dict})

    # Register plain config (no groups) - Hydra only handles global scalars
    ConfigStore.instance().store(name=task_name, node=OmegaConf.create(cfg_dict))
    return env_cfg, agent_cfg, hydra_args


def parse_overrides(args: list[str], presets: dict) -> tuple:
    """Categorize command line args by type.

    Args:
        args: Command line args (without script name)
        presets: {"env": {"path": {"name": cfg}}, "agent": {...}}

    Returns:
        (global_presets, preset_sel, preset_scalar, global_scalar) where:
        - global_presets: [name, ...] - apply to all matching configs
        - preset_sel: [(section, path, name), ...] - REPLACE selections
        - preset_scalar: [(full_path, value), ...] - scalars in preset paths
        - global_scalar: [arg, ...] - pass to Hydra
    """
    preset_paths = {f"{s}.{p}" if p else s for s, v in presets.items() for p in v}
    global_presets, preset_sel, preset_scalar, global_scalar = [], [], [], []

    for arg in args:
        if "=" not in arg:
            global_scalar.append(arg)
            continue
        key, val = arg.split("=", 1)
        if key == "presets":
            known_names = _known_preset_names(presets)
            global_presets.extend(_normalize_preset_name(v.strip(), known_names) for v in val.split(",") if v.strip())
        elif key in preset_paths:
            sec, path = key.split(".", 1) if "." in key else (key, "")
            known_names = set(presets[sec][path])
            preset_sel.append((sec, path, _normalize_preset_name(val, known_names)))
        elif any(key.startswith(pp + ".") for pp in preset_paths):
            preset_scalar.append((key, val))
        else:
            global_scalar.append(arg)

    preset_sel.sort(key=lambda x: x[1].count("."))
    return global_presets, preset_sel, preset_scalar, global_scalar


def apply_overrides(
    env_cfg,
    agent_cfg,
    hydra_cfg: dict,
    global_presets: list,
    preset_sel: list,
    preset_scalar: list,
    presets: dict,
):
    """Apply preset selections and scalar overrides with REPLACE semantics.

    Presets are resolved by walking the active tree from root to leaves. A
    nested preset is only considered after its parent branch has been selected,
    which prevents inactive sibling branches from contributing colliding
    descendant paths.

    Returns:
        (env_cfg, agent_cfg) -- possibly replaced if root-level PresetCfg was resolved.

    Raises:
        ValueError: If multiple global presets conflict on an active path, or
            an explicit preset path is not reachable in the active tree.
    """
    cfgs = {"env": env_cfg, "agent": agent_cfg}

    explicit = {f"{sec}.{path}" if path else sec: name for sec, path, name in preset_sel}
    for sec in ("env", "agent"):
        if cfgs[sec] is None:
            continue
        section_explicit = {path: name for path, name in explicit.items() if path == sec or path.startswith(sec + ".")}
        cfgs[sec] = _resolve_active_presets(cfgs[sec], global_presets, section_explicit, root_path=sec)
        hydra_cfg[sec] = (
            cfgs[sec].to_dict()
            if hasattr(cfgs[sec], "to_dict")
            else dict(cfgs[sec])
            if isinstance(cfgs[sec], Mapping)
            else cfgs[sec]
        )

    _apply_preset_scalars(cfgs, hydra_cfg, preset_scalar)
    return cfgs["env"], cfgs["agent"]


def _apply_preset_scalars(cfgs: dict, hydra_cfg: dict, preset_scalar: list) -> None:
    for full_path, val_str in preset_scalar:
        sec = full_path.split(".", 1)[0]
        if sec not in cfgs:
            continue
        path = full_path[len(sec) + 1 :]
        if cfgs[sec] is not None:
            val = _parse_val(val_str)
            _setattr(cfgs[sec], path, val)
            _setattr(hydra_cfg, full_path, val)


def _setattr(obj, path: str, val):
    """Set nested attribute/key (e.g., "actions.arm_action.scale")."""
    *parts, leaf = path.split(".")
    for p in parts:
        obj = obj[p] if isinstance(obj, Mapping) else getattr(obj, p)
    if isinstance(obj, dict):
        obj[leaf] = val
    else:
        setattr(obj, leaf, val)


def _parse_val(s: str):
    """Parse string to Python value (bool, None, int, float, or str)."""
    if s.lower() in _LITERAL_MAP:
        return _LITERAL_MAP[s.lower()]
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s
