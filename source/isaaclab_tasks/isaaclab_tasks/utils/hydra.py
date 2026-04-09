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
    1. Global presets: ``presets=inference,newton`` -- apply everywhere matching
    2. Path presets: ``env.backend=newton`` -- REPLACE specific section
    3. Preset-path scalars: ``env.backend.dt=0.001`` -- handled by us
    4. Global scalars: ``env.decimation=10`` -- handled by Hydra

Example usage::

    presets=newton env.backend.dt=0.001 env.decimation=10
"""

import functools
import sys
from collections.abc import Callable, Mapping

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError("Hydra not installed. Run: pip install hydra-core")

from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import configclass, replace_slices_with_strings, replace_strings_with_slices

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


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
            newton: NewtonCfg = NewtonCfg()
    """

    pass


def preset(**options) -> PresetCfg:
    """Create a :class:`PresetCfg` instance from keyword arguments.

    A convenience factory that dynamically builds a ``PresetCfg`` subclass
    with one field per keyword argument, then returns an instance of it.
    The caller **must** supply a ``default`` key.

    Example::

        armature = preset(default=0.0, newton=0.01)
        # Equivalent to:
        # @configclass
        # class _Preset(PresetCfg):
        #     default: float = 0.0
        #     newton: float = 0.01
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
        ``{"backend": {"default": PhysxCfg(), "newton": NewtonCfg()}}``
    """
    result = {}

    def _record(preset_obj, preset_path):
        fields = _preset_fields(preset_obj)
        result[preset_path] = fields
        for alt in fields.values():
            if hasattr(alt, "__dataclass_fields__"):
                result.update(collect_presets(alt, preset_path))

    if isinstance(cfg, PresetCfg):
        _record(cfg, path)
        return result

    _walk_cfg(cfg, path, lambda _p, _k, obj, cp: _record(obj, cp))
    return result


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

    Safe to call before Kit is launched — callable config values are stored as
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


def resolve_preset_defaults(cfg):
    """Replace PresetCfg fields with their ``default`` value, recursively.

    Must be called before ``to_dict()`` so the Hydra dict contains only the
    resolved config rather than the raw PresetCfg with all alternatives.
    Returns the (possibly replaced) cfg if the root itself is a PresetCfg.
    """
    if isinstance(cfg, PresetCfg):
        default = getattr(cfg, "default", None)
        return resolve_preset_defaults(default) if default is not None else cfg

    def _on_preset(parent, key, preset_obj, _path):
        default = getattr(preset_obj, "default", None)
        if default is None:
            return
        if isinstance(parent, dict):
            parent[key] = default
        else:
            setattr(parent, key, default)
        if hasattr(default, "__dataclass_fields__"):
            resolve_preset_defaults(default)

    _walk_cfg(cfg, "", _on_preset)
    return cfg


def register_task(task_name: str, agent_entry: str) -> tuple:
    """Load configs, collect presets recursively, register base config to Hydra.

    Presets are collected from nested configclasses and stored separately -
    NOT registered as Hydra groups to avoid Hydra's merge behavior.

    Returns:
        (env_cfg, agent_cfg, presets) where presets =
        {"env": {"path": {"name": cfg}}, "agent": {...}}
    """
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = None
    if agent_entry:
        agent_cfg = load_cfg_from_registry(task_name, agent_entry)

    # Collect presets recursively from the config tree
    presets = {
        "env": collect_presets(env_cfg),
        "agent": collect_presets(agent_cfg) if agent_cfg else {},
    }

    # Resolve PresetCfg defaults before serialization so to_dict() doesn't
    # include all preset alternatives in the hydra dict.
    env_cfg = resolve_preset_defaults(env_cfg)
    if agent_cfg is not None:
        agent_cfg = resolve_preset_defaults(agent_cfg)

    # Convert to dict for Hydra (handle gym spaces and slices)
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    if agent_cfg is not None and hasattr(agent_cfg, "to_dict"):
        agent_dict = agent_cfg.to_dict()
    else:
        agent_dict = agent_cfg
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
    # Build lookup of preset group paths (e.g., "env.actions").
    # Root-level PresetCfg has path="" -> bare "env" or "agent" key.
    preset_paths = {f"{s}.{p}" if p else s for s, v in presets.items() for p in v}
    global_presets, preset_sel, preset_scalar, global_scalar = [], [], [], []

    for arg in args:
        if "=" not in arg:
            global_scalar.append(arg)
            continue
        key, val = arg.split("=", 1)
        if key == "presets":
            global_presets.extend(v.strip() for v in val.split(",") if v.strip())
        elif key in preset_paths:
            sec, path = key.split(".", 1) if "." in key else (key, "")
            preset_sel.append((sec, path, val))
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

    Phase 1: Determine the selected preset name for every path (explicit
    selections, then global broadcasts, then ``default`` fallback).
    Phase 2: Apply in depth order, pruning children whose parent is None.
    Phase 3: Apply scalar overrides on top.

    Returns:
        (env_cfg, agent_cfg) — possibly replaced if root-level PresetCfg was resolved.

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

    def _apply_node(sec: str, path: str, node):
        node_dict = node.to_dict() if hasattr(node, "to_dict") else dict(node) if isinstance(node, Mapping) else node
        if path == "":
            cfgs[sec] = node
            hydra_cfg[sec] = node_dict
        else:
            _setattr(cfgs[sec], path, node)
            _setattr(hydra_cfg, f"{sec}.{path}", node_dict)

    # --- Phase 1: Determine selected preset name for every path ---------------
    resolved: dict[str, tuple[str, str, str]] = {}
    for sec, path, name in preset_sel:
        if path not in presets.get(sec, {}):
            raise ValueError(f"Unknown preset group: {sec}.{path}")
        if name not in presets[sec][path]:
            avail = list(presets[sec][path].keys())
            raise ValueError(f"Unknown preset '{name}' for {sec}.{path}. Available: {avail}")
        full_path = f"{sec}.{path}" if path else sec
        resolved[full_path] = (sec, path, name)

    # Apply global presets (error on real conflict — same path, different value)
    applied_by: dict[str, str] = {}
    for name in global_presets:
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
                    if full_path not in resolved:
                        resolved[full_path] = (sec, path, name)

    # Fill remaining paths with "default" (if available)
    for sec in ("env", "agent"):
        for path, path_presets in presets.get(sec, {}).items():
            full_path = f"{sec}.{path}" if path else sec
            if full_path not in resolved and "default" in path_presets:
                resolved[full_path] = (sec, path, "default")

    # --- Phase 2: Apply in depth order, pruning unreachable children ----------
    for full_path in sorted(resolved, key=lambda fp: fp.count(".")):
        sec, path, name = resolved[full_path]
        if cfgs[sec] is not None and _path_reachable(sec, path):
            _apply_node(sec, path, presets[sec][path][name])

    # 3. Apply scalar overrides within preset paths
    for full_path, val_str in preset_scalar:
        if full_path.startswith("env."):
            sec, path = "env", full_path[4:]
        elif full_path.startswith("agent."):
            sec, path = "agent", full_path[6:]
        else:
            continue
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
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        # Strip quotes if present
        if s[0] in "\"'" and s[-1] in "\"'":
            return s[1:-1]
        return s
