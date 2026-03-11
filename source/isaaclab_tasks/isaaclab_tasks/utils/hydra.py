# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hydra utilities with REPLACE-only preset system.

This module bypasses Hydra's default MERGE behavior for config groups.
Instead, when a preset is selected, the entire config section is REPLACED
with the preset - no field merging.

Presets are defined on individual configclasses using a `presets` dict.
The system recursively discovers all presets and their paths automatically.

Override categories (applied in order):
    1. Global presets: `presets=inference,newton` -> apply everywhere matching
    2. Path presets: `env.actions=joint_control` -> REPLACE specific section
    3. Preset-path scalars: `env.actions.scale=5.0` -> handled by us
    4. Global scalars: `env.sim.dt=0.01` -> handled by Hydra

Example usage:
    # Apply "inference" preset everywhere it exists, then override scale
    presets=inference env.actions.arm_action.scale=0.5 env.sim.dt=0.01
"""

import contextlib
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
from isaaclab_tasks.utils.presets import PresetCfg, UnavailablePreset  # noqa: F401


def collect_presets(cfg, path: str = "") -> dict:
    """Recursively walk config tree and collect presets with auto-discovered paths.

    Each configclass can define a `presets` dict attribute. Presets are collected
    at the path where they're found. The dict can have two formats:

    1. Simple format (presets for the current field):
       presets = {"preset_name": ConfigInstance(), ...}
       -> Collected at current path

    2. Nested format (presets for child fields, legacy support):
       presets = {"child.path": {"preset_name": ConfigInstance()}, ...}
       -> Collected at specified sub-path

    Args:
        cfg: A configclass instance to walk.
        path: Current path prefix (used during recursion).

    Returns:
        Dict mapping paths to preset dicts, e.g.:
        {"actions.arm_action": {"relative_joint_position": <cfg>, ...}}
    """
    result = {}

    # Root-level PresetCfg: the cfg itself is a PresetCfg subclass
    if isinstance(cfg, PresetCfg) and hasattr(cfg, "__dataclass_fields__"):
        if getattr(cfg, "presets", None):
            raise ValueError(
                f"PresetCfg subclass at '{path or '<root>'}' must not define a 'presets' attribute. "
                "Use typed fields instead (e.g., default: MyCfg = MyCfg())."
            )
        preset_dict = {}
        for field_name in cfg.__dataclass_fields__:
            preset_dict[field_name] = getattr(cfg, field_name)
        result[path] = preset_dict
        for alt in preset_dict.values():
            if hasattr(alt, "__dataclass_fields__"):
                result.update(collect_presets(alt, path))
        return result

    # Check if this config has presets
    presets = getattr(cfg, "presets", None)
    if presets and isinstance(presets, dict):
        # Check format: simple (values are configs) or nested (values are dicts)
        first_val = next(iter(presets.values()), None) if presets else None
        is_nested_format = isinstance(first_val, dict)

        if is_nested_format:
            # Nested format: {"field.path": {"name": cfg}}
            for sub_path, sub_presets in presets.items():
                full_path = f"{path}.{sub_path}" if path else sub_path
                result[full_path] = sub_presets
        else:
            # Simple format: {"name": cfg} - presets for current field
            if path:
                result[path] = presets

    # Recurse into nested configclass attributes
    for name in dir(cfg):
        if name.startswith("_") or name == "presets":
            continue
        try:
            value = getattr(cfg, name)
        except Exception:
            continue

        # Check if it's a configclass (has __dataclass_fields__)
        if hasattr(value, "__dataclass_fields__"):
            child_path = f"{path}.{name}" if path else name
            if isinstance(value, PresetCfg):
                if getattr(value, "presets", None):
                    raise ValueError(
                        f"PresetCfg subclass at '{child_path}' must not define a 'presets' attribute. "
                        "Use typed fields instead (e.g., default: MyCfg = MyCfg())."
                    )
                preset_dict = {}
                for field_name in value.__dataclass_fields__:
                    preset_dict[field_name] = getattr(value, field_name)
                result[child_path] = preset_dict
                # Recurse into each alternative to find nested PresetCfg
                for alt in preset_dict.values():
                    if hasattr(alt, "__dataclass_fields__"):
                        result.update(collect_presets(alt, child_path))
            else:
                result.update(collect_presets(value, child_path))

    return result


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

    global_presets, preset_sel, preset_scalar, global_scalar = parse_overrides(sys.argv[1:], presets)

    original_argv, sys.argv = sys.argv, [sys.argv[0]] + global_scalar

    resolved = {}

    @hydra.main(config_path=None, config_name=task, version_base="1.3")
    def hydra_main(hydra_cfg, env_cfg=env_cfg, agent_cfg=agent_cfg):
        hydra_cfg = replace_strings_with_slices(OmegaConf.to_container(hydra_cfg, resolve=True))
        env_cfg, agent_cfg = apply_overrides(
            env_cfg, agent_cfg, hydra_cfg, global_presets, preset_sel, preset_scalar, presets
        )
        _cleanup_presets(env_cfg)
        _cleanup_presets(agent_cfg)
        _cleanup_presets_dict(hydra_cfg)
        env_cfg.from_dict(hydra_cfg["env"])
        env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
        if isinstance(agent_cfg, dict) or agent_cfg is None:
            agent_cfg = hydra_cfg["agent"]
        else:
            agent_cfg.from_dict(hydra_cfg["agent"])
        resolved["env_cfg"] = env_cfg
        resolved["agent_cfg"] = agent_cfg

    try:
        hydra_main()
    finally:
        sys.argv = original_argv

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

            # Split args: global presets, path presets, scalars
            global_presets, preset_sel, preset_scalar, global_scalar = parse_overrides(sys.argv[1:], presets)

            # Only pass global scalars to Hydra
            original_argv, sys.argv = sys.argv, [sys.argv[0]] + global_scalar

            @hydra.main(config_path=None, config_name=task, version_base="1.3")
            def hydra_main(hydra_cfg, env_cfg=env_cfg, agent_cfg=agent_cfg):
                hydra_cfg = replace_strings_with_slices(OmegaConf.to_container(hydra_cfg, resolve=True))
                env_cfg, agent_cfg = apply_overrides(
                    env_cfg, agent_cfg, hydra_cfg, global_presets, preset_sel, preset_scalar, presets
                )
                _cleanup_presets(env_cfg)
                _cleanup_presets(agent_cfg)
                _cleanup_presets_dict(hydra_cfg)
                env_cfg.from_dict(hydra_cfg["env"])
                env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    agent_cfg = hydra_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_cfg["agent"])
                func(env_cfg, agent_cfg, *args, **kwargs)

            try:
                hydra_main()
            finally:
                sys.argv = original_argv

        return wrapper

    return decorator


def resolve_preset_defaults(cfg):
    """Replace PresetCfg fields with their 'default' value before serialization.

    This must be called before to_dict() so the hydra dict contains only the
    resolved config rather than the raw PresetCfg with all alternatives.
    Returns the (possibly replaced) cfg if the root itself is a PresetCfg.
    """
    if isinstance(cfg, PresetCfg) and hasattr(cfg, "__dataclass_fields__"):
        default = getattr(cfg, "default", None)
        if default is not None and not isinstance(default, UnavailablePreset):
            return resolve_preset_defaults(default)
        return cfg

    for name in list(getattr(cfg, "__dataclass_fields__", {}).keys()):
        try:
            value = getattr(cfg, name)
        except Exception:
            continue
        if isinstance(value, PresetCfg) and hasattr(value, "__dataclass_fields__"):
            default = getattr(value, "default", None)
            if default is not None and not isinstance(default, UnavailablePreset):
                setattr(cfg, name, default)
                resolve_preset_defaults(default)
        elif hasattr(value, "__dataclass_fields__"):
            resolve_preset_defaults(value)
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
    # Build lookup of preset group paths (e.g., "env.actions")
    # Root-level PresetCfg has path="" -> bare "env" or "agent" key
    preset_paths = {f"{s}.{p}" if p else s for s, v in presets.items() for p in v}
    global_presets, preset_sel, preset_scalar, global_scalar = [], [], [], []

    for arg in args:
        if "=" not in arg:
            global_scalar.append(arg)
            continue

        key, val = arg.split("=", 1)
        if key == "presets":
            # Global presets: presets=name1,name2 -> apply everywhere
            global_presets.extend(v.strip() for v in val.split(",") if v.strip())
        elif key in preset_paths:
            # Exact match -> preset selection
            if "." in key:
                sec, path = key.split(".", 1)
            else:
                sec, path = key, ""
            preset_sel.append((sec, path, val))
        elif any(key.startswith(pp + ".") for pp in preset_paths):
            # Prefix match -> scalar within preset path
            preset_scalar.append((key, val))
        else:
            # No match -> global scalar, let Hydra handle it
            global_scalar.append(arg)

    # Sort preset selections: parents before children
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

    This is the core function that implements REPLACE (not merge) behavior:
    0. Auto-apply "default" presets for paths not explicitly selected
    1. Global presets (presets=name) apply to ALL configs that have matching preset
    2. Path-specific presets (env.actions=name) replace specific sections
    3. Scalar overrides are applied on top

    Returns:
        (env_cfg, agent_cfg) — possibly replaced if root-level PresetCfg was resolved.

    Raises:
        ValueError: If multiple global presets conflict on the same path.
    """
    cfgs = {"env": env_cfg, "agent": agent_cfg}

    def _path_reachable(sec: str, path: str) -> bool:
        """Check that every ancestor along *path* is non-None on the live config.

        For "scene.camera" we walk to ``cfg.scene`` and verify it is not None,
        then check ``cfg.scene.camera`` is not None.  If any segment is missing
        or None the child preset cannot be applied.
        """
        if not path:
            return cfgs[sec] is not None
        obj = cfgs[sec]
        for part in path.split("."):
            try:
                obj = getattr(obj, part)
            except (AttributeError, TypeError):
                return False
            if obj is None:
                return False
        return True

    def _apply_node(sec: str, path: str, node):
        """Replace a config node at the given section/path, handling root (path='')."""
        if node is None:
            node_dict = None
        elif hasattr(node, "to_dict"):
            node_dict = node.to_dict()
        else:
            node_dict = dict(node)
        if path == "":
            cfgs[sec] = node
            hydra_cfg[sec] = node_dict
        else:
            _setattr(cfgs[sec], path, node)
            _setattr(hydra_cfg, f"{sec}.{path}", node_dict)

    # --- Phase 1: Determine selected preset name for every path ---------------
    # Start with explicit path selections
    resolved: dict[str, tuple[str, str, str]] = {}  # full_path -> (sec, path, name)
    for sec, path, name in preset_sel:
        if path not in presets.get(sec, {}):
            raise ValueError(f"Unknown preset group: {sec}.{path}")
        if name not in presets[sec][path]:
            avail = list(presets[sec][path].keys())
            raise ValueError(f"Unknown preset '{name}' for {sec}.{path}. Available: {avail}")
        full_path = f"{sec}.{path}" if path else sec
        resolved[full_path] = (sec, path, name)

    # Apply global presets (error on conflict)
    applied_by: dict[str, str] = {}
    for name in global_presets:
        for sec in ("env", "agent"):
            for path, path_presets in presets.get(sec, {}).items():
                if name in path_presets:
                    full_path = f"{sec}.{path}" if path else sec
                    if full_path in applied_by:
                        raise ValueError(
                            f"Conflicting global presets: '{applied_by[full_path]}' and '{name}' "
                            f"both define preset for '{full_path}'"
                        )
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
            node = presets[sec][path][name]
            # UnavailablePreset sentinel means the backend package is not installed.
            if isinstance(node, UnavailablePreset):
                raise ValueError(
                    f"Preset '{name}' is not available because its backend package is not"
                    f" installed. To fix this, run: {node.install_cmd}"
                )
            _apply_node(sec, path, node)

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


def _cleanup_presets(cfg):
    """Recursively remove ``presets`` fields from a config tree after resolution.

    Once all presets have been selected and applied, the ``presets`` metadata
    is no longer needed and would confuse downstream managers that expect every
    field to be a term config.
    """
    if cfg is None:
        return
    if hasattr(cfg, "presets"):
        with contextlib.suppress(Exception):
            delattr(cfg, "presets")
    for name in list(getattr(cfg, "__dataclass_fields__", {}).keys()):
        if name == "presets":
            continue
        try:
            value = getattr(cfg, name)
        except Exception:
            continue
        if hasattr(value, "__dataclass_fields__"):
            _cleanup_presets(value)


def _cleanup_presets_dict(d: dict):
    """Recursively remove ``presets`` keys from a nested dict."""
    if not isinstance(d, dict):
        return
    d.pop("presets", None)
    for v in d.values():
        if isinstance(v, dict):
            _cleanup_presets_dict(v)


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
