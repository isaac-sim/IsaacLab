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

import argparse
import ast
import enum
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import is_dataclass
from types import SimpleNamespace
from typing import ClassVar

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from isaaclab.app import validate_launch
from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import (
    configclass,
    replace_slices_with_strings,
    replace_strings_with_slices,
    to_dict,
    update_from_dict,
    validate,
)
from isaaclab.visualizers import VisualizerBackendCfg, VisualizerCfg
from isaaclab.visualizers.visualizer_cfg import migrate_visualizer_cfgs

_LITERAL_MAP = {"true": True, "false": False, "none": None, "null": None}


_LEGACY_PRESET_ALIASES = {
    "newton": "newton_mjwarp",
    "kamino": "newton_kamino",
    "ovrtx_renderer": "ovrtx",
    "isaacsim_rtx_renderer": "isaacsim_rtx",
}


class PresetTarget(enum.Enum):
    """Classify preset alternatives; untyped values belong to the domain category."""

    PHYSICS = ("physics", (PhysicsCfg,))
    RENDERER = ("renderer", (RendererCfg,))
    VISUALIZER = ("visualizer", (VisualizerBackendCfg,))
    DOMAIN = ("presets", ())

    def __new__(cls, label: str, base_classes: tuple[type, ...]):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.base_classes = base_classes
        return obj

    def matches(self, cfg: object) -> bool:
        """Match a config type, including physics bundled with simulation-wide settings.

        Args:
            cfg: Preset alternative to classify.

        Returns:
            Whether the alternative belongs to this category.
        """
        return isinstance(cfg, self.base_classes) or (
            self is PresetTarget.PHYSICS and isinstance(cfg, SimulationCfg) and isinstance(cfg.physics, PhysicsCfg)
        )


def user_stacklevel() -> int:
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
    while frame is not None and frame.f_globals.get("__name__", "").startswith("isaaclab_tasks.utils"):
        level += 1
        frame = frame.f_back
        if level > max_walk:
            return 2
    return level


def _normalize_preset_name(name: str, known_names: set[str], aliases: Mapping[str, str] | None = None) -> str:
    """Map a deprecated preset name to its replacement and emit a warning.

    Returns ``name`` unchanged when:
        * ``name`` is not a deprecated alias, or
        * the replacement is not declared in ``known_names`` (so the user-supplied
          value can flow into the unknown-preset error), or
        * ``name`` is itself a real field in ``known_names`` (a user-defined preset
          legitimately reusing the deprecated spelling shadows the alias).
    """
    replacement = (aliases or {}).get(name, _LEGACY_PRESET_ALIASES.get(name))
    if replacement is None or replacement not in known_names or name in known_names:
        return name
    warnings.warn(
        f"Preset '{name}' is deprecated. Use '{replacement}' instead.",
        FutureWarning,
        stacklevel=user_stacklevel(),
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

    _aliases: ClassVar[dict[str, str]] = {}

    def __getattr__(self, name: str):
        """Alias a deprecated preset name to its replacement field.

        Raises ``AttributeError`` for any other missing attribute so that
        ``hasattr`` and standard introspection keep working unchanged. The
        replacement is only returned when the deprecated name is *not* itself a
        real field on the subclass, so a user redefining the deprecated name
        shadows the alias.
        """
        replacement = self._aliases.get(name, _LEGACY_PRESET_ALIASES.get(name))
        fields = self.__dataclass_fields__
        if replacement is not None and replacement in fields and name not in fields:
            warnings.warn(
                f"Preset '{name}' is deprecated. Use '{replacement}' instead.",
                FutureWarning,
                stacklevel=user_stacklevel(),
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
        if fn.startswith("_"):
            continue
        cls_val = getattr(cls, fn, None)
        d[fn] = cls_val if cls_val is not None else getattr(preset_obj, fn)
    for attr in vars(cls):
        if attr.startswith("_") or attr in d or callable(getattr(cls, attr)):
            continue
        d[attr] = getattr(cls, attr)
    return d


def _iter_cfg_items(cfg):
    """Iterate declared config fields and container entries without inspecting properties."""
    if is_dataclass(cfg) and not isinstance(cfg, type) or isinstance(cfg, (argparse.Namespace, SimpleNamespace)):
        return ((name, value) for name, value in vars(cfg).items() if not name.startswith("_"))
    if isinstance(cfg, Mapping):
        return cfg.items()
    if isinstance(cfg, (list, tuple)):
        return enumerate(cfg)
    return ()


def _has_interpolations(cfg: object) -> bool:
    """Check whether the active configuration contains OmegaConf interpolations."""
    if isinstance(cfg, str):
        return "${" in cfg
    return any(_has_interpolations(value) for _, value in _iter_cfg_items(cfg))


def collect_presets(cfg, path: str = "") -> dict:
    """Discover preset alternatives, including those inside inactive branches.

    Args:
        cfg: Configuration tree to inspect.
        path: Prefix for the returned dotted paths.

    Returns:
        A mapping from config paths to named alternatives.
    """
    result = {}
    if isinstance(cfg, PresetCfg):
        result[path] = _preset_fields(cfg)
        for alternative in result[path].values():
            result.update(collect_presets(alternative, path))
    else:
        for key, value in _iter_cfg_items(cfg):
            result.update(collect_presets(value, f"{path}.{key}" if path else str(key)))
    return result


# ============================================================================
# Preset resolution
# ============================================================================


def _pick_alternative(
    preset_obj: PresetCfg,
    selected,
    path: str = "",
    explicit_name: str | Sequence[str] | None = None,
    consumed_selected: set[str] | None = None,
    on_selected: Callable[[str, object], None] | None = None,
):
    """Choose the best alternative from a PresetCfg.

    Priority: first match in ``selected``, then ``default`` (preferring
    class-level over instance-level).

    Raises:
        ValueError: If no matching name and no ``default`` field exists.
    """
    if explicit_name is not None and not isinstance(explicit_name, str):
        choices = [
            _pick_alternative(preset_obj, selected, path, name, consumed_selected, on_selected)
            for name in explicit_name
        ]
        if len(choices) > 1 and any(choice is None or (isinstance(choice, list) and not choice) for choice in choices):
            raise ValueError("An empty preset choice cannot be combined with other choices.")
        return choices[0] if len(choices) == 1 else choices
    fields = _preset_fields(preset_obj)
    field_names = set(fields)
    if explicit_name is not None:
        explicit_name = _normalize_preset_name(explicit_name, field_names, preset_obj._aliases)
        if explicit_name in fields:
            return fields[explicit_name]
        avail = list(fields)
        hint = ""
        if explicit_name in _LEGACY_PRESET_ALIASES:
            replacement = _LEGACY_PRESET_ALIASES[explicit_name]
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
        if on_selected is not None:
            on_selected(raw_name, val)
            if name != raw_name:
                on_selected(name, val)
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
    explicit: dict[str, str | Sequence[str]] | None = None,
    *,
    strict_explicit: bool = True,
    consumed_selected: set[str] | None = None,
    on_selected: Callable[[str, object], None] | None = None,
    consumed_explicit: set[str] | None = None,
    selectors: Mapping[Callable[[object], bool], str | Sequence[str]] | None = None,
    consumed_selectors: set[Callable] | None = None,
):
    """Resolve presets by walking only the currently active tree.

    Only the selected alternative contributes descendant presets.
    """
    explicit = explicit or {}
    consumed_explicit = consumed_explicit if consumed_explicit is not None else set()
    consumed_selectors = consumed_selectors if consumed_selectors is not None else set()

    def resolve(value, path):
        seen = set()
        while isinstance(value, PresetCfg):
            if id(value) in seen:
                raise ValueError(
                    f"Cyclic PresetCfg chain detected at '{path}': {type(value).__name__} was already visited."
                )
            seen.add(id(value))
            choice = explicit.get(path)
            if choice is not None:
                consumed_explicit.add(path)
            for matches, names in (selectors or {}).items():
                if any(matches(alternative) for alternative in _preset_fields(value).values()):
                    if choice is None:
                        choice = names
                    consumed_selectors.add(matches)
            value = _pick_alternative(value, selected, path or "<root>", choice, consumed_selected, on_selected)
        if isinstance(value, tuple):
            return tuple(resolve(child, f"{path}.{key}" if path else str(key)) for key, child in enumerate(value))
        for key, child in _iter_cfg_items(value):
            resolved = resolve(child, f"{path}.{key}" if path else str(key))
            if resolved is child:
                continue
            if isinstance(value, (dict, list)):
                value[key] = resolved
            else:
                setattr(value, key, resolved)
        return value

    cfg = resolve(cfg, "")
    missing = sorted(set(explicit) - consumed_explicit)
    if strict_explicit and missing:
        raise ValueError(f"Unknown or inactive preset group(s): {', '.join(missing)}")
    return cfg


def resolve_presets(
    cfg,
    selected=(),
    *,
    overrides: dict[str, str | Sequence[str]] | None = None,
):
    """Replace every :class:`PresetCfg` in the tree with the best alternative.

    For each ``PresetCfg`` found in the active configuration tree:

    1. Choose a matching name from *selected*, rejecting conflicting alternatives,
       or use ``default`` if none match.
    2. Replace the preset in its parent (dict key or dataclass attr).
    3. Continue walking the replacement (which may contain more presets).

    Args:
        cfg: A configclass, dict, or PresetCfg to resolve in-place.
        selected: Set of preset names chosen by the user (e.g. from CLI
            ``presets=peg_insert_4mm,eval``).
        overrides: Path-specific choices. A list of names selects a list of configurations.

    Returns:
        The resolved ``cfg`` (possibly a different object if the root itself
        was a PresetCfg).
    """
    return _resolve_active_presets(cfg, selected, explicit=overrides)


# ============================================================================
# CLI / Hydra integration
# ============================================================================


def resolve_task_config(
    task_name: str,
    agent_cfg_entry_point: str | None,
    play_mode: bool = False,
    overrides: Sequence[str] | None = None,
):
    """Resolve env and agent configs with Hydra overrides, presets, scalars, and interpolations applied.

    Safe to call before Kit is launched -- callable config values are stored as
    :class:`~isaaclab.utils.string.ResolvableString` and resolved lazily on
    first use, so no implementation modules are imported eagerly.

    Args:
        task_name: Task name (e.g., "IsaacContrib-Velocity-Flat-AnymalC").
        agent_cfg_entry_point: Agent config entry point key (e.g., "rsl_rl_cfg_entry_point").
        play_mode: Whether to apply the play-mode overrides defined by the environment
            configuration's ``play_mode`` method after loading. Defaults to False.
        overrides: Optional Hydra arguments to use instead of reading them from
            :data:`sys.argv`. This keeps programmatic task composition on the same
            path as command-line composition. Defaults to None.

    Returns:
        Tuple of (env_cfg, agent_cfg) fully resolved.
    """
    task = task_name.split(":")[-1]
    env_cfg, agent_cfg, hydra_args = register_task(
        task, agent_cfg_entry_point, play_mode=play_mode, overrides=overrides
    )
    if not hydra_args:
        return replace_strings_with_env_cfg_spaces(env_cfg), agent_cfg

    resolved = {}
    original_argv, sys.argv = sys.argv, [sys.argv[0]] + hydra_args

    @hydra.main(config_path=None, config_name=task)
    def hydra_main(hydra_cfg, env_cfg=env_cfg, agent_cfg=agent_cfg):
        env_cfg, agent_cfg = _apply_resolved_config(env_cfg, agent_cfg, hydra_cfg)
        resolved.update(env_cfg=replace_strings_with_env_cfg_spaces(env_cfg), agent_cfg=agent_cfg)

    try:
        hydra_main()
    finally:
        sys.argv = original_argv
    return resolved["env_cfg"], resolved["agent_cfg"]


def register_task(
    task_name: str,
    agent_entry: str | None,
    play_mode: bool = False,
    overrides: Sequence[str] | None = None,
) -> tuple:
    """Load configs, collect presets recursively, register base config to Hydra.

    Presets are collected from nested configclasses and stored separately -
    NOT registered as Hydra groups to avoid Hydra's merge behavior.

    Args:
        task_name: Task name (e.g., "Isaac-Reach-Franka").
        agent_entry: Agent config entry point key.
        play_mode: Whether to apply the play-mode overrides defined by the environment
            configuration's ``play_mode`` method after loading. Defaults to False.
        overrides: Optional Hydra arguments to compose instead of :data:`sys.argv`.
            Non-string literals replace resolved preset values; strings at preset paths select alternatives.

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
    selectors = {}
    consumed_selectors = set()
    override_items: list[tuple[str, str, str]] = []
    hydra_args: list[str] = []
    for arg in sys.argv[1:] if overrides is None else overrides:
        if "=" not in arg:
            hydra_args.append(arg)
            continue
        key, val = arg.split("=", 1)
        token = key.lstrip("-")
        if token == PresetTarget.DOMAIN.value:
            global_presets.extend(v.strip() for v in val.split(",") if v.strip())
        elif token == PresetTarget.VISUALIZER.value:
            names = [name.strip() for name in val.split(",") if name.strip()]
            if not names:
                raise ValueError("visualizer= requires a preset name; use visualizer=none to disable viewers.")
            selectors[PresetTarget.VISUALIZER.matches] = names if len(names) > 1 else names[0]
        elif token in typed_labels:
            for name in (v.strip() for v in val.split(",") if v.strip()):
                global_presets.append(name)
                requested_targets.setdefault(typed_labels[token], set()).add(name)
        else:
            override_items.append((key, val, arg))

    # Literal values replace the resolved value after presets and play-mode overrides.
    explicit = {key: val for key, val, _arg in override_items if isinstance(_parse_val(val), str)}
    consumed_presets: set[str] = set()
    typed_hits: dict[str, set[PresetTarget]] = {}

    def record_selection(name, value):
        targets = {target for target in PresetTarget if target.base_classes and target.matches(value)}
        typed_hits.setdefault(name, set()).update(targets)

    consumed_explicit: set[str] = set()
    cfgs = {"env": env_cfg, "agent": agent_cfg}
    _resolve_active_presets(
        cfgs,
        global_presets,
        explicit,
        strict_explicit=False,
        consumed_selected=consumed_presets,
        on_selected=record_selection,
        consumed_explicit=consumed_explicit,
        selectors=selectors,
        consumed_selectors=consumed_selectors,
    )
    env_cfg, agent_cfg = cfgs["env"], cfgs["agent"]

    unknown_presets = set(global_presets) - consumed_presets
    if unknown_presets:
        # Build the full discovery table only on the error path, or when a
        # selected name applies only to inactive branches and therefore has no
        # effect in the active-tree walk.
        all_presets = {
            "env": collect_presets(load_cfg_from_registry(task_name, "env_cfg_entry_point")),
            "agent": collect_presets(load_cfg_from_registry(task_name, agent_entry)) if agent_entry else {},
        }
        available = {
            f"{section}.{path}" if path else section: fields
            for section, presets in all_presets.items()
            for path, fields in presets.items()
        }
        known_names = {name for fields in available.values() for name in fields}
        normalized_names = {name: _normalize_preset_name(name, known_names) for name in unknown_presets}
        unknown = set(normalized_names.values()) - known_names
        unavailable_typed_names = {
            normalized_names[name]
            for target, names in requested_targets.items()
            if not any(
                target.matches(value)
                for fields in available.values()
                for preset_name, value in fields.items()
                if preset_name != "default"
            )
            for name in names
            if name in normalized_names
        }
        unknown -= unavailable_typed_names
        if unknown:
            lines = [f"Unknown preset(s): {', '.join(sorted(unknown))}"]
            for name in sorted(unknown & _LEGACY_PRESET_ALIASES.keys()):
                lines.append(f"'{name}' was renamed to '{_LEGACY_PRESET_ALIASES[name]}'; this task declares neither.")
            lines.append("Available presets by config path:")
            lines.extend(
                f"  {path}: {', '.join(name for name in fields if name != 'default')}"
                for path, fields in available.items()
            )
            raise ValueError("\n".join(lines))

    missing = [
        f"{target.value}={name}"
        for target, names in requested_targets.items()
        for name in sorted(names)
        if target not in typed_hits.get(_LEGACY_PRESET_ALIASES.get(name, name), ())
    ]
    if missing:
        raise ValueError(
            f"Typed preset selector(s) {', '.join(missing)} did not match a selectable preset of that type for task "
            f"'{task_name}'. Choose a declared preset, or remove the selector(s) to use its registered default "
            "configuration."
        )
    if PresetTarget.VISUALIZER.matches in selectors.keys() - consumed_selectors:
        from .presets import MultiBackendVisualizerCfg

        env_cfg.sim.visualizer_cfgs = migrate_visualizer_cfgs(
            env_cfg.sim.visualizer_cfgs, env_cfg.sim.default_visualizer_cfg
        )
        env_cfg.sim.default_visualizer_cfg = None
        viewers = env_cfg.sim.visualizer_cfgs
        if not viewers:
            viewers = [VisualizerCfg()]
        for viewer in viewers:
            choices = MultiBackendVisualizerCfg()
            backends = viewer.backend if isinstance(viewer.backend, list) else [viewer.backend]
            for backend in backends:
                vars(choices)[backend.visualizer_type] = backend
            viewer.backend = resolve_presets(choices, overrides={"": selectors[PresetTarget.VISUALIZER.matches]})
        env_cfg.sim.visualizer_cfgs = viewers

    # apply play-mode overrides after preset resolution so they act on the resolved
    # config, and before scalar overrides so explicit user values still win
    if play_mode and hasattr(env_cfg, "play_mode"):
        env_cfg.play_mode()

    for key, val, arg in override_items:
        value = _parse_val(val)
        if key in consumed_explicit and isinstance(value, str):
            continue
        if key.startswith(("env.", "agent.")) and not key.endswith("+"):
            sec, path = key.split(".", 1)
            _setattr(cfgs[sec], path, value)
        else:
            hydra_args.append(arg)

    if not hydra_args and not _has_interpolations(cfgs):
        return env_cfg, agent_cfg, hydra_args

    # Convert to dict for Hydra (handle gym spaces and slices)
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    agent_dict = agent_cfg.to_dict() if agent_cfg is not None and hasattr(agent_cfg, "to_dict") else agent_cfg
    env_dict = to_dict(env_cfg)  # type: ignore[union-attr]
    cfg_dict = replace_slices_with_strings({"env": env_dict, "agent": agent_dict})
    hydra_cfg = OmegaConf.create(cfg_dict)

    if not hydra_args:
        env_cfg, agent_cfg = _apply_resolved_config(env_cfg, agent_cfg, hydra_cfg)
        return env_cfg, agent_cfg, hydra_args

    # Register plain config (no groups) - Hydra only handles global scalars
    ConfigStore.instance().store(name=task_name, node=hydra_cfg, replace=True)
    return env_cfg, agent_cfg, hydra_args


def _apply_resolved_config(env_cfg, agent_cfg, hydra_cfg) -> tuple:
    """Apply resolved values while preserving environment and agent configclass instances."""
    cfg_dict = replace_strings_with_slices(OmegaConf.to_container(hydra_cfg, resolve=True))
    update_from_dict(env_cfg, cfg_dict["env"])
    if isinstance(agent_cfg, dict) or agent_cfg is None:
        agent_cfg = cfg_dict["agent"]
    else:
        update_from_dict(agent_cfg, cfg_dict["agent"])
    return env_cfg, agent_cfg


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


def setup_preset_cli(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> tuple[argparse.Namespace, list[str]]:
    """Register the preset-selection help description and parse argv.

    Must be called *after* launcher flags and script-specific arguments are
    registered on ``parser`` -- otherwise those unknown tokens land in
    ``parse_known_args``'s remainder.

    The remainder contains canonical preset selectors, Hydra path overrides,
    and unknown flags. Launcher arguments, including ``--kit_args``, are parsed
    first so their values are never interpreted as task selectors.

    Does not mutate ``sys.argv``; the caller assigns
    ``sys.argv = [sys.argv[0]] + remaining`` when ready, so any argv-aware logic
    that re-reads ``sys.argv`` (e.g. an external callback) runs against the
    user's original command line first.

    Args:
        parser: Caller's argument parser. An ``argument_group`` is attached
            for help-time variant discovery. No preset selector arguments are
            added, so the Namespace gains no preset attributes.
        argv: Arguments to parse, including help-time task selection. Defaults to ``sys.argv[1:]``.
    Returns:
        Parsed launcher arguments and normalized task overrides, ready to hand
        to Hydra via ``sys.argv`` or :func:`~isaaclab_tasks.utils.hydra.resolve_task_config`.

    Raises:
        SystemExit: If help is requested or selector arguments are missing or conflicting.
    """
    args_to_parse = sys.argv[1:] if argv is None else argv
    help_requested = "-h" in args_to_parse or "--help" in args_to_parse
    variants = None
    if help_requested:
        # Read the task before the main parser's help action exits.
        task_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        task_parser.add_argument("--task")
        task_args, _ = task_parser.parse_known_args(args_to_parse)
        if task_args.task:
            variants = _enumerate_variants(task_args.task)
    if parser.formatter_class is argparse.HelpFormatter:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
    description = ["Select named PresetCfg alternatives via Hydra-style overrides (key=value, no leading dashes):"]
    for target in PresetTarget:
        syntax = f"{target.value}=NAME"
        if target in (PresetTarget.VISUALIZER, PresetTarget.DOMAIN):
            syntax += "[,NAME,...]"
        meaning = (
            f"(typed) selects a {target.value} backend"
            if target.base_classes
            else "broadcast: applied to every matching PresetCfg"
        )
        description.append(f"    {syntax:23}   {meaning}" + (". Available:" if variants is not None else ""))
        if variants is not None:
            description.extend(f"                              - {name}" for name in variants[target])
            if not variants[target]:
                description.append("                              (none)")
    if variants is None:
        description.append("\nPass `--task=X` along with `--help` to see preset variants available for that task.")
    description.append(
        "\nAliases: --physics NAME, --renderer NAME, --visualizer NAME (--viz).\n"
        "Hydra also accepts path-targeted overrides like env.sim.physics=NAME."
    )
    parser.add_argument_group("preset selection", description="\n".join(description))
    if help_requested:
        parser.print_help()
        raise SystemExit(0)

    optional_value_consts = {
        option: action.const
        for action in parser._actions
        if action.nargs == argparse.OPTIONAL
        for option in action.option_strings
    }
    args_to_parse = [
        f"{token}={optional_value_consts[token]}" if token in optional_value_consts and "=" in next_token else token
        for token, next_token in zip(args_to_parse, [*args_to_parse[1:], ""])
    ]
    args, remaining = parser.parse_known_args(args_to_parse)
    overrides = []
    tokens = iter(remaining)
    for token in tokens:
        key, separator, value = token.partition("=")
        if key in ("--physics", "--renderer", "--visualizer", "--viz"):
            value = value if separator else next(tokens, "")
            if not value or value.startswith("-") or "=" in value:
                parser.error(f"{key} requires a preset name.")
            key = "visualizer" if key == "--viz" else key[2:]
            token = f"{key}={value}"
        overrides.append(token)

    for key in ("physics", "renderer", "visualizer"):
        selected = {token[len(key) + 1 :] for token in overrides if token.startswith(f"{key}=")}
        value = vars(args).pop(key, None)
        explicit = vars(args).pop(f"{key}_explicit", False)
        if value is not None or explicit:
            value = (",".join(value) if isinstance(value, list) else value) or "none"
            if (explicit or not selected) and value not in selected:
                overrides.append(f"{key}={value}")
                selected.add(value)
        if len(selected) > 1:
            parser.error(f"Conflicting {key} selections: {', '.join(sorted(selected))}.")
    return args, overrides


def enumerate_task_presets(task_name: str) -> dict[PresetTarget, list[str]] | None:
    """Return the available preset names for *task_name*, bucketed by selector type.

    Loads the env config registered under *task_name* and walks its preset tree
    using the same logic that the CLI help-text renderer uses, so the returned
    view matches what ``--task=<name> --help`` shows at the command line.

    This function is safe to call after :class:`~isaaclab_physx.app.KitLauncher` has
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
        return _enumerate_variants(task_name)
    except Exception:
        return None


def _enumerate_variants(task_name: str) -> dict[PresetTarget, list[str]]:
    """Load env_cfg for *task_name* and bucket its variants by target.

    Uses the same walker hydra's resolver runs so help and resolve see one
    view of the cfg tree. The env_cfg load is safe before the simulation runtime starts
    because ``test_env_cfg_no_forbidden_imports`` blocks Kit-only imports at
    the top level of cfg modules. Exceptions from the loader propagate
    verbatim -- they surface as the natural error, not a buried help string.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    typed_targets = [t for t in PresetTarget if t.base_classes]
    result: dict[PresetTarget, set[str]] = {target: set() for target in PresetTarget}
    for path_dict in collect_presets(env_cfg).values():
        families = {target for target in typed_targets if any(target.matches(cfg) for cfg in path_dict.values())}
        for name, cfg in path_dict.items():
            if name == "default":
                continue
            matched = next((target for target in typed_targets if target.matches(cfg)), PresetTarget.DOMAIN)
            if (cfg is None or (isinstance(cfg, list) and not cfg)) and len(families) == 1:
                matched = next(iter(families))
            result[matched].add(name)
    if not result[PresetTarget.VISUALIZER]:
        from .presets import MultiBackendVisualizerCfg

        result[PresetTarget.VISUALIZER].update(
            name for name in collect_presets(MultiBackendVisualizerCfg())[""] if name != "default"
        )
    return {target: sorted(names) for target, names in result.items()}


_SELECTORS = ("physics", "renderer", "presets")


def task_run_overrides(run: tuple[str | None, str | None, str | None]) -> list[str]:
    """Return the command-line tokens of a ``(physics, renderer, presets)`` run, e.g. ``["physics=ovphysx"]``."""
    return [f"{selector}={name}" for selector, name in zip(_SELECTORS, run) if name is not None]


def _launchable_env_cfg(task_name: str, run: tuple[str | None, str | None, str | None]):
    """Return the resolved and launch-scanned env config of *run*, or ``None`` if it is rejected."""
    try:
        env_cfg, _ = resolve_task_config(task_name, None, overrides=task_run_overrides(run))
        validate(env_cfg)
        validate_launch(env_cfg, {"livestream": 0})
    except ValueError:
        return None
    return env_cfg


def is_task_run_launchable(task_name: str, run: tuple[str | None, str | None, str | None]) -> bool:
    """Return whether a ``(physics, renderer, presets)`` run of *task_name* resolves, passes the env config's
    checks, and can launch headless.

    A rejected combination is one whose resolution or validation raises :class:`ValueError`; any other
    error propagates.
    """
    return _launchable_env_cfg(task_name, run) is not None


def resolve_task_runs(task_name: str) -> list[tuple[str | None, str | None, str | None]] | None:
    """Return the preset combinations of *task_name* that can launch, one per distinct config.

    Each physics and renderer preset, or leaving that axis to the config, is crossed with each domain
    preset in turn. Every combination is resolved, validated with the env config's own checks, and
    checked with :func:`~isaaclab.app.validate_launch` for a headless launch without livestreaming. Combinations
    that resolve to the same config, such as ``physics=physx`` and the backend it selects, are
    reported once, by their most explicit spelling.

    Only one domain preset is applied per run, so a task with several independent domain
    presets supports more combinations than are listed. A rejected combination is one whose
    resolution or validation raises :class:`ValueError`; any other error propagates.

    Args:
        task_name: Gymnasium task ID (e.g. ``"Isaac-Cartpole-Camera-Direct"``).

    Returns:
        ``(physics, renderer, presets)`` names per run, ``None`` on an axis left to the config. The
        first run is the one the task uses when no names are passed, unless that run cannot launch.
        ``None`` if the task's env config cannot be loaded.
    """
    preset_map = enumerate_task_presets(task_name)
    if preset_map is None:
        return None
    typed = {*preset_map[PresetTarget.PHYSICS], *preset_map[PresetTarget.RENDERER]}
    domains = [name for name in preset_map[PresetTarget.DOMAIN] if name not in typed]
    candidates = [
        (physics, renderer, domain)
        for physics in [None, *preset_map[PresetTarget.PHYSICS]]
        for renderer in [None, *preset_map[PresetTarget.RENDERER]]
        for domain in [None, *domains]
    ]

    runs: dict[str, tuple[str | None, str | None, str | None]] = {}
    default_key = None
    for run in candidates:
        env_cfg = _launchable_env_cfg(task_name, run)
        if env_cfg is None:
            continue
        # Fingerprint after launch validation resolves automatic selectors to concrete configs.
        key = repr(to_dict(env_cfg))
        if run == (None, None, None):
            default_key = key
        if key not in runs or len(task_run_overrides(run)) > len(task_run_overrides(runs[key])):
            runs[key] = run
    default = [runs.pop(default_key)] if default_key is not None else []
    return default + list(runs.values())
