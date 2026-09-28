# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config-only preset declarations and selection, shared by tasks and standalone applications."""

import sys
import warnings
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar

from .configclass import configclass

_LEGACY_PRESET_ALIASES = {
    "newton": "newton_mjwarp",
    "kamino": "newton_kamino",
    "ovrtx_renderer": "ovrtx",
    "isaacsim_rtx_renderer": "isaacsim_rtx",
}


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
    while frame is not None and frame.f_globals.get("__name__", "").startswith(
        ("isaaclab.utils.presets", "isaaclab_tasks.utils")
    ):
        level += 1
        frame = frame.f_back
        if level > max_walk:
            return 2
    return level


def _known_preset_names(presets: dict) -> set[str]:
    """Return all preset names declared in a collected preset dictionary."""
    return {name for section in presets.values() for fields in section.values() for name in fields}


def _normalize_preset_name(name: str, known_names: set[str], aliases: Mapping[str, str] | None = None) -> str:
    """Map a deprecated preset name to its replacement and emit a warning.

    Returns ``name`` unchanged when:
        * ``name`` is not a deprecated alias, or
        * the replacement is not declared in ``known_names`` (so the user-supplied
          value can flow into the standard "unknown preset" error path, where
          :func:`_format_unknown_presets_error` will surface the rename), or
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
        fields = getattr(type(self), "__dataclass_fields__", {})
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
    root_path: str = "",
    *,
    strict_explicit: bool = True,
    consumed_selected: set[str] | None = None,
    on_selected: Callable[[str, object], None] | None = None,
    consumed_explicit: set[str] | None = None,
    selectors: Mapping[Callable[[object], bool], str | Sequence[str]] | None = None,
    consumed_selectors: set[Callable] | None = None,
):
    """Resolve presets by walking only the currently active tree.

    Preset alternatives are choice nodes. Once a choice is resolved, only the
    selected replacement is queued for further traversal, so inactive sibling
    branches cannot contribute descendant presets.
    """
    explicit = explicit or {}
    consumed_explicit = consumed_explicit if consumed_explicit is not None else set()
    consumed_selectors = consumed_selectors if consumed_selectors is not None else set()

    def resolve_chain(preset_obj: PresetCfg, path: str):
        seen: set[int] = set()
        val = preset_obj
        while isinstance(val, PresetCfg):
            if id(val) in seen:
                raise ValueError(
                    f"Cyclic PresetCfg chain detected at '{path}': {type(val).__name__} was already visited."
                )
            seen.add(id(val))
            choice = explicit.get(path)
            if choice is None:
                alternatives = _preset_fields(val).values()
                for matches, names in (selectors or {}).items():
                    if any(matches(value) for value in alternatives):
                        choice = names
                        consumed_selectors.add(matches)
            val = _pick_alternative(
                val,
                selected,
                path=path or "<root>",
                explicit_name=choice,
                consumed_selected=consumed_selected,
                on_selected=on_selected,
            )
        return val

    if isinstance(cfg, PresetCfg):
        if root_path in explicit:
            consumed_explicit.add(root_path)
        cfg = resolve_chain(cfg, root_path)

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


def resolve_presets(
    cfg,
    selected=(),
    *,
    overrides: dict[str, str | Sequence[str]] | None = None,
    selectors: Mapping[Callable[[object], bool], str | Sequence[str]] | None = None,
):
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
        overrides: Path-specific choices. A list of names selects a list of configurations.
        selectors: Typed choices, keyed by a predicate matching the alternatives.

    Returns:
        The resolved ``cfg`` (possibly a different object if the root itself
        was a PresetCfg).
    """
    consumed = set()
    cfg = _resolve_active_presets(cfg, selected, explicit=overrides, selectors=selectors, consumed_selectors=consumed)
    if selectors and selectors.keys() - consumed:
        raise ValueError("The configuration does not declare a preset for the requested selector.")
    return cfg
