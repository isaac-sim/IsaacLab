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
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


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
            result.update(collect_presets(value, child_path))

    return result


def hydra_task_config(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Decorator for Hydra config with REPLACE-only preset semantics.

    Args:
        task_name: Task name (e.g., "Isaac-Reach-Franka-v0")
        agent_cfg_entry_point: Agent config entry point key

    Returns:
        Decorated function receiving (env_cfg, agent_cfg, *args, **kwargs)
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
                # Apply our preset handling (REPLACE, not merge)
                apply_overrides(env_cfg, agent_cfg, hydra_cfg, global_presets, preset_sel, preset_scalar, presets)
                # Sync dict -> config objects
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
    preset_paths = {f"{s}.{p}" for s, v in presets.items() for p in v}
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
            sec, path = key.split(".", 1)
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

    Raises:
        ValueError: If multiple global presets conflict on the same path.
    """
    cfgs = {"env": env_cfg, "agent": agent_cfg}

    # Build set of paths that will be explicitly selected
    selected_paths = {f"{sec}.{path}" for sec, path, _ in preset_sel}
    for name in global_presets:
        for sec in ("env", "agent"):
            for path, path_presets in presets.get(sec, {}).items():
                if name in path_presets:
                    selected_paths.add(f"{sec}.{path}")

    # 0. Auto-apply "default" presets for paths not explicitly selected
    for sec in ("env", "agent"):
        for path, path_presets in presets.get(sec, {}).items():
            full_path = f"{sec}.{path}"
            if full_path not in selected_paths and "default" in path_presets:
                node = path_presets["default"]
                if cfgs[sec]:
                    _setattr(cfgs[sec], path, node)
                    if hasattr(node, "to_dict"):
                        node_dict = node.to_dict()
                    else:
                        node_dict = dict(node)
                    _setattr(hydra_cfg, full_path, node_dict)

    # 1. Apply global presets - find all paths with matching preset name
    # Track which paths are set by which preset to detect conflicts
    applied_by: dict[str, str] = {}  # full_path -> preset_name
    for name in global_presets:
        for sec in ("env", "agent"):
            for path, path_presets in presets.get(sec, {}).items():
                if name in path_presets:
                    full_path = f"{sec}.{path}"
                    if full_path in applied_by:
                        raise ValueError(
                            f"Conflicting global presets: '{applied_by[full_path]}' and '{name}' "
                            f"both define preset for '{full_path}'"
                        )
                    applied_by[full_path] = name
                    node = path_presets[name]
                    if cfgs[sec]:
                        _setattr(cfgs[sec], path, node)
                        if hasattr(node, "to_dict"):
                            node_dict = node.to_dict()
                        else:
                            node_dict = dict(node)
                        _setattr(hydra_cfg, f"{sec}.{path}", node_dict)

    # 2. Apply path-specific preset selections (REPLACE entire section)
    for sec, path, name in preset_sel:
        if path not in presets.get(sec, {}):
            raise ValueError(f"Unknown preset group: {sec}.{path}")
        if name not in presets[sec][path]:
            avail = list(presets[sec][path].keys())
            raise ValueError(f"Unknown preset '{name}' for {sec}.{path}. Available: {avail}")
        node = presets[sec][path][name]
        if cfgs[sec]:
            # REPLACE on config object
            _setattr(cfgs[sec], path, node)
            # REPLACE on dict (for from_dict sync)
            if hasattr(node, "to_dict"):
                node_dict = node.to_dict()
            else:
                node_dict = dict(node)
            _setattr(hydra_cfg, f"{sec}.{path}", node_dict)

    # 2. Apply scalar overrides within preset paths
    for full_path, val_str in preset_scalar:
        if full_path.startswith("env."):
            sec, path = "env", full_path[4:]
        elif full_path.startswith("agent."):
            sec, path = "agent", full_path[6:]
        else:
            continue
        if cfgs[sec]:
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
