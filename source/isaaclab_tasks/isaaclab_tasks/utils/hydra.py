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
    2. Path presets: ``env.backend=newton_mjwarp`` -- REPLACE specific section
    3. Preset-path scalars: ``env.backend.dt=0.001`` -- handled by us
    4. Global scalars: ``env.decimation=10`` -- handled by Hydra

Example usage::

    presets=newton_mjwarp env.backend.dt=0.001 env.decimation=10
"""

import functools
import sys
import warnings
from collections.abc import Callable, Mapping

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import configclass, replace_slices_with_strings, replace_strings_with_slices

_LITERAL_MAP = {"true": True, "false": False, "none": None, "null": None}

# Map of deprecated preset name -> current name. Newton-backend solver presets are
# now prefixed with ``newton_`` so they group together in autocomplete and read
# distinctly from backend/package/visualizer names that also use the word
# ``newton``. Aliases keep legacy CLI invocations and ``PresetCfg`` field accesses
# working with a :class:`FutureWarning`; they will be removed in a future release.
_LEGACY_PRESET_ALIASES = {"newton": "newton_mjwarp", "kamino": "newton_kamino"}


def _user_stacklevel() -> int:
    """Compute a ``warnings.warn`` stacklevel that lands on the first frame
    outside this module, so deprecation messages cite user code rather than
    internal hydra-utility frames.

    Walks at most a small bounded number of frames; if no non-hydra frame is
    found within the bound (frozen modules, exec'd contexts, or oddly named
    ``__file__`` globals), falls back to ``stacklevel=2`` so the warning at
    least jumps out of the helper that called it.
    """
    max_walk = 16
    level = 1
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__file__") == __file__:
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
    replacement = _LEGACY_PRESET_ALIASES.get(name)
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
    """

    def __getattr__(self, name: str):
        """Alias a deprecated preset name to its replacement field.

        Raises ``AttributeError`` for any other missing attribute so that
        ``hasattr`` and standard introspection keep working unchanged. The
        replacement is only returned when the deprecated name is *not* itself a
        real field on the subclass, so a user redefining the deprecated name
        shadows the alias.
        """
        replacement = _LEGACY_PRESET_ALIASES.get(name)
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


def _walk_cfg(cfg, path: str, on_preset: Callable) -> None:
    """Depth-first walk of a config tree, calling *on_preset(parent, key, obj, path)*
    for every :class:`PresetCfg` node.  Recurses through dataclass attrs, dicts, and
    nested dicts transparently."""
    items = (
        cfg.items()
        if isinstance(cfg, dict)
        else ((n, v) for n in dir(cfg) if not n.startswith("_") for v in [getattr(cfg, n, None)] if v is not None)
    )
    for key, val in items:
        child_path = f"{path}.{key}" if path else key
        if isinstance(val, PresetCfg):
            on_preset(cfg, key, val, child_path)
        elif hasattr(val, "__dataclass_fields__") or isinstance(val, dict):
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
                    if hasattr(v, "__dataclass_fields__"):
                        result.update(collect_presets(v, preset_path))

    if isinstance(cfg, PresetCfg):
        _record(cfg, path)
        return result

    _walk_cfg(cfg, path, lambda _p, _k, obj, cp: _record(obj, cp))
    return result


# ============================================================================
# Preset resolution
# ============================================================================


def _pick_alternative(preset_obj: PresetCfg, selected: set[str], path: str = ""):
    """Choose the best alternative from a PresetCfg.

    Priority: first match in ``selected``, then ``default`` (preferring
    class-level over instance-level).

    Raises:
        ValueError: If no matching name and no ``default`` field exists.
    """
    fields = _preset_fields(preset_obj)
    field_names = set(fields)
    for name in selected:
        name = _normalize_preset_name(name, field_names)
        if name in fields:
            return fields[name]
    if "default" in fields:
        return fields["default"]
    raise ValueError(
        f"PresetCfg {type(preset_obj).__name__} at '{path}' has no 'default' field "
        f"and none of the selected presets {selected} match its fields {set(fields.keys())}."
    )


def resolve_presets(cfg, selected: set[str] = frozenset()):
    """Replace every :class:`PresetCfg` in the tree with the best alternative.

    For each ``PresetCfg`` found during a depth-first walk:

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
    if isinstance(cfg, PresetCfg):
        seen: set[int] = {id(cfg)}
        replacement = _pick_alternative(cfg, selected, path="<root>")
        while isinstance(replacement, PresetCfg):
            if id(replacement) in seen:
                raise ValueError(
                    f"Cyclic PresetCfg chain detected at '<root>': {type(replacement).__name__} was already visited."
                )
            seen.add(id(replacement))
            replacement = _pick_alternative(replacement, selected, path="<root>")
        return resolve_presets(replacement, selected)

    def _resolve(parent, key, preset_obj, _path):
        seen: set[int] = {id(preset_obj)}
        val = _pick_alternative(preset_obj, selected, path=_path)
        while isinstance(val, PresetCfg):
            if id(val) in seen:
                raise ValueError(
                    f"Cyclic PresetCfg chain detected at '{_path}': {type(val).__name__} was already visited."
                )
            seen.add(id(val))
            val = _pick_alternative(val, selected, path=_path)
        if isinstance(parent, dict):
            parent[key] = val
        else:
            setattr(parent, key, val)
        if hasattr(val, "__dataclass_fields__") or isinstance(val, dict):
            _walk_cfg(val, _path, _resolve)

    _walk_cfg(cfg, "", _resolve)
    return cfg


# ============================================================================
# CLI / Hydra integration
# ============================================================================


def _run_hydra(task, env_cfg, agent_cfg, presets, callback):
    """Shared Hydra entry point for :func:`resolve_task_config` and :func:`hydra_task_config`."""
    global_presets, preset_sel, preset_scalar, global_scalar = parse_overrides(sys.argv[1:], presets)
    original_argv, sys.argv = sys.argv, [sys.argv[0]] + global_scalar

    @hydra.main(config_path=None, config_name=task, version_base="1.3")
    def hydra_main(hydra_cfg, env_cfg=env_cfg, agent_cfg=agent_cfg):
        hydra_cfg = replace_strings_with_slices(OmegaConf.to_container(hydra_cfg, resolve=True))
        env_cfg, agent_cfg = apply_overrides(
            env_cfg, agent_cfg, hydra_cfg, global_presets, preset_sel, preset_scalar, presets
        )
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
    env_cfg, agent_cfg, presets = register_task(task, agent_cfg_entry_point)
    resolved = {}
    _run_hydra(task, env_cfg, agent_cfg, presets, lambda e, a: resolved.update(env_cfg=e, agent_cfg=a))
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
            env_cfg, agent_cfg, presets = register_task(task, agent_cfg_entry_point)
            _run_hydra(task, env_cfg, agent_cfg, presets, lambda e, a: func(e, a, *args, **kwargs))

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
    deprecated_hits = sorted(name for name in unknown if name in _LEGACY_PRESET_ALIASES)
    for legacy in deprecated_hits:
        replacement = _LEGACY_PRESET_ALIASES[legacy]
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


def register_task(task_name: str, agent_entry: str) -> tuple:
    """Load configs, collect presets recursively, register base config to Hydra.

    Presets are collected from nested configclasses and stored separately -
    NOT registered as Hydra groups to avoid Hydra's merge behavior.

    Returns:
        (env_cfg, agent_cfg, presets) where presets =
        {"env": {"path": {"name": cfg}}, "agent": {...}}
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_name, agent_entry) if agent_entry else None

    # Collect presets before resolution (needed for path-based overrides)
    presets = {
        "env": collect_presets(env_cfg),
        "agent": collect_presets(agent_cfg) if agent_cfg else {},
    }

    known_names = _known_preset_names(presets)
    selected = {
        _normalize_preset_name(v.strip(), known_names)
        for arg in sys.argv[1:]
        if "=" in arg
        for key, val in [arg.split("=", 1)]
        if key.lstrip("-") == "presets"
        for v in val.split(",")
        if v.strip()
    }

    if selected:
        name_to_paths: dict[str, list[str]] = {}
        for sec, sec_presets in presets.items():
            for path, fields in sec_presets.items():
                full = f"{sec}.{path}" if path else sec
                for name in fields:
                    name_to_paths.setdefault(name, []).append(full)
        unknown = selected - set(name_to_paths)
        if unknown:
            display = {n: p for n, p in name_to_paths.items() if n != "default"}
            raise ValueError(_format_unknown_presets_error(unknown, display))

    env_cfg = resolve_presets(env_cfg, selected)
    if agent_cfg is not None:
        agent_cfg = resolve_presets(agent_cfg, selected)

    # Also resolve presets inside collected alternatives so that apply_overrides
    # never re-introduces unresolved PresetCfg objects when applying a selection.
    for section_presets in presets.values():
        for path_presets in section_presets.values():
            for name, alt in path_presets.items():
                resolve_presets(alt, selected)

    # Convert to dict for Hydra (handle gym spaces and slices)
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    agent_dict = agent_cfg.to_dict() if agent_cfg is not None and hasattr(agent_cfg, "to_dict") else agent_cfg
    env_dict = env_cfg.to_dict()  # type: ignore[union-attr]
    cfg_dict = replace_slices_with_strings({"env": env_dict, "agent": agent_dict})

    # Register plain config (no groups) - Hydra only handles global scalars
    ConfigStore.instance().store(name=task_name, node=OmegaConf.create(cfg_dict))
    return env_cfg, agent_cfg, presets


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

    Global presets are already applied by :func:`resolve_presets` in
    :func:`register_task`. This function handles:

    1. Path-based selections (``env.backend=newton_mjwarp``)
    2. Scalar overrides within preset paths (``env.backend.dt=0.001``)

    Returns:
        (env_cfg, agent_cfg) -- possibly replaced if root-level PresetCfg was resolved.

    Raises:
        ValueError: If multiple global presets conflict on the same path.
    """
    cfgs = {"env": env_cfg, "agent": agent_cfg}

    def _path_reachable(sec: str, path: str) -> bool:
        if not path:
            return cfgs[sec] is not None
        obj = cfgs[sec]
        for part in path.split("."):
            try:
                obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
            except (AttributeError, TypeError, KeyError):
                return False
            if obj is None:
                return False
        return True

    # --- Phase 1: path-based selections + global broadcast for reachable paths
    resolved: dict[str, tuple[str, str, str]] = {}
    for sec, path, name in preset_sel:
        if path not in presets.get(sec, {}):
            raise ValueError(f"Unknown preset group: {sec}.{path}")
        name = _normalize_preset_name(name, set(presets[sec][path]))
        if name not in presets[sec][path]:
            avail = list(presets[sec][path].keys())
            hint = ""
            if name in _LEGACY_PRESET_ALIASES:
                replacement = _LEGACY_PRESET_ALIASES[name]
                hint = f" '{name}' was renamed to '{replacement}'; this path does not declare '{replacement}' either."
            raise ValueError(f"Unknown preset '{name}' for {sec}.{path}. Available: {avail}.{hint}")
        full_path = f"{sec}.{path}" if path else sec
        resolved[full_path] = (sec, path, name)

    applied_by: dict[str, str] = {}
    known_names = _known_preset_names(presets)
    for name in global_presets:
        name = _normalize_preset_name(name, known_names)
        for sec in ("env", "agent"):
            for path, path_presets in presets.get(sec, {}).items():
                if name in path_presets:
                    full_path = f"{sec}.{path}" if path else sec
                    if full_path in applied_by:
                        prev_name = applied_by[full_path]
                        prev_val = path_presets[prev_name]
                        cur_val = path_presets[name]
                        if prev_val is not cur_val and prev_val != cur_val:
                            raise ValueError(
                                f"Conflicting global presets: '{prev_name}' and '{name}' "
                                f"both define preset for '{full_path}'"
                            )
                    else:
                        applied_by[full_path] = name
                    resolved.setdefault(full_path, (sec, path, name))

    for sec in ("env", "agent"):
        for path, path_presets in presets.get(sec, {}).items():
            if "default" in path_presets:
                full_path = f"{sec}.{path}" if path else sec
                resolved.setdefault(full_path, (sec, path, "default"))

    # --- Phase 2: apply in depth order, pruning unreachable children
    for full_path in sorted(resolved, key=lambda fp: fp.count(".")):
        sec, path, name = resolved[full_path]
        if cfgs[sec] is not None and _path_reachable(sec, path):
            node = presets[sec][path][name]
            node_dict = (
                node.to_dict() if hasattr(node, "to_dict") else dict(node) if isinstance(node, Mapping) else node
            )
            if not path:
                cfgs[sec], hydra_cfg[sec] = node, node_dict
            else:
                _setattr(cfgs[sec], path, node)
                _setattr(hydra_cfg, f"{sec}.{path}", node_dict)

    # --- Phase 3: scalar overrides within preset paths
    for full_path, val_str in preset_scalar:
        sec = full_path.split(".", 1)[0]
        if sec not in cfgs:
            continue
        path = full_path[len(sec) + 1 :]
        if cfgs[sec] is not None:
            val = _parse_val(val_str)
            _setattr(cfgs[sec], path, val)
            _setattr(hydra_cfg, full_path, val)

    return cfgs["env"], cfgs["agent"]


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
        return float(s) if "." in s else int(s)
    except ValueError:
        return s[1:-1] if len(s) >= 2 and s[0] in "\"'" and s[-1] in "\"'" else s
