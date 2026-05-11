# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical preset name registry.

Two things live here:

* :class:`PresetTarget` -- the closed enum of preset categories. Each
  member carries its CLI-flag label and its (optional) per-target dict
  of legacy aliases. Adding a category = appending one enum member.
* :class:`PresetRegistry` -- the ``{target: {name: cls}}`` container plus
  ``register`` decorator and lookups. All state + access on one class.

The module-level :func:`register` alias keeps the natural decorator
spelling at the call site.

Example::

    from isaaclab.utils.preset_registry import PresetTarget, register


    @register(PresetTarget.PHYSICS, "physx")
    @configclass
    class PhysxCfg(PhysicsCfg): ...

This module lives in :mod:`isaaclab.utils` so backend packages
(``isaaclab_physx``, ``isaaclab_newton``, ...) can decorate their cfg
classes without taking a dependency on :mod:`isaaclab_tasks`.
"""

from __future__ import annotations

import enum
import warnings
from typing import ClassVar


class PresetTarget(enum.Enum):
    """CLI-flag target categories.

    Each member's value is ``(label, legacy_aliases)``:

    * ``label`` -- the lowercase CLI flag string. ``--{label}`` becomes
      the typed flag for non-DOMAIN targets; ``DOMAIN`` is the catch-all
      that maps to ``--presets`` and is never validated.
    * ``legacy_aliases`` -- mapping of deprecated preset names to their
      canonical replacements within this target. Optional; targets with
      no legacy names omit it.

    Adding a new target = appending one enum member; ``setup_cli`` and
    ``PresetRegistry`` discover it via iteration. No second list to update.
    """

    # Members. Tuple values are (label, legacy_aliases). The enum metaclass
    # collects the whole namespace before constructing members, so ``__new__``
    # below picks these up regardless of declaration order.
    PHYSICS = ("physics", {"newton": "newton_mjwarp", "kamino": "newton_kamino"})
    """Physics backends -- ``--physics`` flag. Legacy: ``newton``, ``kamino``."""

    RENDERER = ("renderer",)
    """Camera-sensor renderers -- ``--renderer`` flag."""

    DOMAIN = ("domain",)
    """Free-form env-specific presets -- ``--presets`` flag (catch-all). Not validated."""

    @classmethod
    def all_legacy_aliases(cls) -> dict[str, str]:
        """Aggregate every target's legacy alias map into one flat dict.

        Resolver-layer code looks up aliases by name without target
        context (the ``presets=...`` token is target-agnostic on the
        wire), so it needs the flat view. Builds fresh from the
        per-target tables on every call to keep PresetTarget the
        single source of truth and avoid a second cached copy.

        Returns:
            Deprecated-name to canonical-replacement mapping aggregated
            from every member's ``legacy_aliases``.
        """
        return {name: rep for target in cls for name, rep in target.legacy_aliases.items()}

    def __new__(cls, label: str, legacy_aliases: dict[str, str] | None = None):
        """Construct a :class:`PresetTarget` member from its tuple value.

        Called by the enum metaclass for every member assignment so the
        tuple ``(label, legacy_aliases)`` unpacks into a label-valued
        member that also carries its own ``legacy_aliases`` mapping.

        Args:
            label: Lowercase CLI flag suffix (e.g. ``"physics"`` becomes
                the ``--physics`` flag and ``self.value``).
            legacy_aliases: Deprecated-to-canonical replacements that
                :meth:`normalize` consults for this target. A fresh copy
                is stored so members cannot alias each other's tables;
                omit (``None``) when the target has no legacy aliases.

        Returns:
            A new enum member with ``_value_`` set to *label* and a
            private ``legacy_aliases`` dict ready for :meth:`normalize`.
        """
        obj = object.__new__(cls)
        obj._value_ = label
        # Per-instance attribute so it survives the enum machinery.
        obj.legacy_aliases = dict(legacy_aliases) if legacy_aliases else {}
        return obj

    def normalize(self, name: str) -> str:
        """Resolve a legacy alias for this target to its canonical name.

        Side effect: a :class:`FutureWarning` is emitted whenever an
        alias is rewritten so users notice the deprecation; non-alias
        names pass through silently.

        Args:
            name: Whatever the user wrote on the typed flag, before
                registry validation.

        Returns:
            *name* unchanged when it is not a legacy alias of this
            target, or the registered canonical replacement when it is.
        """
        if name in self.legacy_aliases:
            canonical = self.legacy_aliases[name]
            # stacklevel=4 = warn() -> normalize() -> _validate_typed_flag()
            # -> setup_cli() -> user's train.py. Lands the warning on the
            # user's setup_cli(...) call instead of inside this module.
            warnings.warn(
                f"--{self.value} {name!r} is deprecated. Use {canonical!r} instead.",
                FutureWarning,
                stacklevel=4,
            )
            return canonical
        return name


class PresetRegistry:
    """``(target, name) → class`` map. Container + register + lookups on one class.

    Populated at backend-cfg import time by the :meth:`register` decorator.
    The module-level :data:`register` alias is the canonical decorator
    call form: ``@register(PresetTarget.PHYSICS, "physx")``.
    """

    # {target: {name: cls}}. ClassVar so it's class-level state, not per-instance.
    _entries: ClassVar[dict[PresetTarget, dict[str, type]]] = {}

    @classmethod
    def register(cls, target: PresetTarget, name: str):
        """Decorator: bind ``(target, name)`` to a config class.

        The decorated class gains ``_preset_name`` (str) and
        ``_preset_target`` (PresetTarget) attributes for later lookup.

        Stamping is per-class -- the guard checks ``target_cls.__dict__``
        only -- so:

        * **Chained decoration of the same class** (rare, usually a
          mistake) preserves the *first* binding's canonical attributes.
          Decorators apply bottom-up, so the inner ``@register`` runs
          first, sets the attribute, and the outer ``@register`` sees
          ``__dict__`` already populated and skips stamping. The outer
          name is still added to the registry so it resolves, but
          ``cls._preset_name`` reads back to the inner one.
        * **Decorated subclass** gets its *own* canonical -- the
          subclass ``__dict__`` doesn't yet contain ``_preset_name``
          (parent's value is reachable via MRO but not via ``__dict__``),
          so the stamp succeeds. Decorating a subclass with a different
          name is a deliberate "new preset" declaration and shadows the
          parent canonical at the subclass level.

        Args:
            target: Which preset target the canonical name belongs to.
                Determines which ``--flag`` will accept it and which
                lookup table the binding lives in.
            name: Canonical name the backend wants to expose. Must be
                unique within *target* across the whole process.

        Returns:
            A class decorator. Applied to a config class, it records
            the binding (so the name resolves via
            :meth:`names_for`) and stamps the class with
            ``_preset_name`` / ``_preset_target`` for MRO lookups.

        Raises:
            RuntimeError: ``(target, name)`` is already bound to a
                different class. Catches accidental name reuse across
                backend packages.
        """

        def deco(target_cls: type) -> type:
            existing = cls._entries.get(target, {}).get(name)
            if existing is not None and existing is not target_cls:
                raise RuntimeError(
                    f"@register({target!r}, {name!r}) already bound to"
                    f" {existing.__module__}.{existing.__name__}; cannot rebind to"
                    f" {target_cls.__module__}.{target_cls.__name__}."
                )
            cls._entries.setdefault(target, {})[name] = target_cls
            # Stamp only when this exact class doesn't already carry a
            # canonical of its own. Chained decoration: first one wins.
            # Decorated subclass: gets its own (parent's value is inherited
            # but not in this class's __dict__).
            if "_preset_name" not in target_cls.__dict__:
                target_cls._preset_name = name  # type: ignore[attr-defined]
                target_cls._preset_target = target  # type: ignore[attr-defined]
            return target_cls

        return deco

    @classmethod
    def names_for(cls, target: PresetTarget) -> set[str]:
        """Look up the canonical names currently bound under *target*.

        Args:
            target: Which target's bindings to read. Note that backends
                only land in the registry after their cfg module has
                been imported; expect an empty set in a fresh process
                before any task or backend has been touched.

        Returns:
            A fresh set the caller may mutate without disturbing the
            registry. Empty when nothing is registered yet for
            *target*.
        """
        return set(cls._entries.get(target, ()))

    @staticmethod
    def canonical_and_target(value: object) -> tuple[str | None, PresetTarget | None]:
        """Identify which ``@register``-stamped class *value* belongs to.

        Walks the MRO of ``type(value)`` for the ``_preset_name`` /
        ``_preset_target`` stamps; falls back to ``value.solver_cfg``'s
        MRO so wrappers like ``NewtonCfg`` (which holds a registered
        solver-cfg) still resolve. Lives on :class:`PresetRegistry`
        because it's fundamentally a registry-side lookup: "which of my
        bindings does this value belong to."

        Args:
            value: An alternative held by a ``PresetCfg`` (a config
                instance, a wrapper exposing ``solver_cfg``, or any
                opaque object that may not be registered at all).

        Returns:
            The canonical name and target paired with *value*'s
            registered class. ``(None, None)`` when neither the MRO nor
            the ``solver_cfg`` fallback hits a registered class.
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


# Decorator alias kept at module level for the natural decorator spelling.
register = PresetRegistry.register
"""Decorator alias for :meth:`PresetRegistry.register`."""


# ----------------------------------------------------------------------------
# Design follow-up (not pursued in this PR)
# ----------------------------------------------------------------------------
# ``DOMAIN`` is structurally a different kind of target than ``PHYSICS`` /
# ``RENDERER``: it's free-form rather than validated, never decorated with
# ``@register``, and has no legacy aliases. The current ``setup_cli`` branches
# on ``if target is PresetTarget.DOMAIN`` in three places (arg registration,
# typed-values collection, name collection). A polymorphic refactor would lift
# those branches into per-kind classes held as enum values::
#
#     class TargetKind:
#         def add_argument(self, group, valid_names, task): ...
#         def collect_names(self, args, variants) -> list[str]: ...
#
#     class TypedTarget(TargetKind):    # PHYSICS / RENDERER
#         def __init__(self, label, legacy_aliases=None): ...
#
#     class DomainTarget(TargetKind):   # DOMAIN -- free-form CSV
#         label = "presets"
#
#     class PresetTarget(enum.Enum):
#         PHYSICS  = TypedTarget("physics", {"newton": "newton_mjwarp", ...})
#         RENDERER = TypedTarget("renderer")
#         DOMAIN   = DomainTarget()
#         def add_argument(self, *a, **kw): return self.value.add_argument(*a, **kw)
#         def collect_names(self, *a, **kw): return self.value.collect_names(*a, **kw)
#
# Net change is roughly line-neutral: ``_help_text`` and ``_validate_typed_flag``
# fold into ``TypedTarget`` methods, the three ``if DOMAIN`` branches disappear,
# and ``@register``'s signature can narrow to reject anything but a typed kind.
# Worth revisiting when a third kind appears (logger / teleop / curriculum)
# whose behavior doesn't fit either of the two existing shapes.
