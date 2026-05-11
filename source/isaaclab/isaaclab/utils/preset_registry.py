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

    def __new__(cls, label: str, legacy_aliases: dict[str, str] | None = None):
        obj = object.__new__(cls)
        obj._value_ = label
        # Per-instance attribute so it survives the enum machinery.
        obj.legacy_aliases = dict(legacy_aliases) if legacy_aliases else {}
        return obj

    def normalize(self, name: str) -> str:
        """Resolve a legacy alias for this target to its canonical name.

        Returns *name* unchanged if it is not a legacy alias of this target.
        Otherwise emits a :class:`FutureWarning` and returns the canonical
        replacement.
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

        Raises:
            RuntimeError: If ``(target, name)`` is already bound to a different class.
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
        """Canonical names registered for *target*."""
        return set(cls._entries.get(target, ()))


# Decorator alias kept at module level for the natural decorator spelling.
register = PresetRegistry.register
"""Decorator alias for :meth:`PresetRegistry.register`."""
