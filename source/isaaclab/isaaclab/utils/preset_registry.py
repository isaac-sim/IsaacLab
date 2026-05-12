# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical preset name registry for the typed-flag CLI layer.

This module exists so backend packages (``isaaclab_physx``,
``isaaclab_newton``, ...) can declare their preset names once via a
decorator and the CLI layer can list them in ``--help`` and accept them
as typed flags. The actual preset semantics (PresetCfg, resolution,
alias rewriting, post-load validation) all live in
:mod:`isaaclab_tasks.utils.hydra`; this is purely a discoverability
layer on top.

Example::

    from isaaclab.utils.preset_registry import PresetTarget, register


    @register(PresetTarget.PHYSICS, "physx")
    @configclass
    class PhysxCfg(PhysicsCfg): ...

This module lives in :mod:`isaaclab.utils` (core) so backend packages
can decorate their cfg classes without taking a dependency on
:mod:`isaaclab_tasks`.
"""

from __future__ import annotations

import enum
from typing import ClassVar


class PresetTarget(enum.Enum):
    """CLI-flag target categories.

    Each member's value is ``(label, legacy_aliases)``:

    * ``label`` -- the lowercase CLI flag string. ``--{label}`` becomes
      the typed flag for non-DOMAIN targets; ``DOMAIN`` is the catch-all
      that maps to ``--presets`` (free-form CSV).
    * ``legacy_aliases`` -- mapping of deprecated preset names to their
      canonical replacements within this target. Optional; targets with
      no legacy names omit it. Hydra's resolver consults these (via
      :func:`isaaclab_tasks.utils.hydra._normalize_preset_name`); the
      data lives here so the typed-flag layer and the resolver share
      one source of truth.

    Adding a new target = appending one enum member.
    """

    # Members. Tuple values are (label, legacy_aliases). The enum metaclass
    # collects the whole namespace before constructing members, so ``__new__``
    # below picks these up regardless of declaration order.
    PHYSICS = ("physics", {"newton": "newton_mjwarp", "kamino": "newton_kamino"})
    """Physics backends -- ``--physics`` flag. Legacy aliases: ``newton``, ``kamino``."""

    RENDERER = ("renderer",)
    """Camera-sensor renderers -- ``--renderer`` flag."""

    DOMAIN = ("domain",)
    """Free-form env-specific presets -- ``--presets`` flag (catch-all). Not validated."""

    def __new__(cls, label: str, legacy_aliases: dict[str, str] | None = None):
        """Construct a member from its ``(label, legacy_aliases)`` tuple.

        Args:
            label: Lowercase CLI flag suffix (e.g. ``"physics"`` becomes
                ``--physics`` and ``self.value``).
            legacy_aliases: Optional deprecated-to-canonical map; copied
                so members cannot alias each other's tables.

        Returns:
            A new enum member with ``_value_`` set to *label* and a
            private ``legacy_aliases`` attribute.
        """
        obj = object.__new__(cls)
        obj._value_ = label
        obj.legacy_aliases = dict(legacy_aliases) if legacy_aliases else {}
        return obj

    @classmethod
    def all_legacy_aliases(cls) -> dict[str, str]:
        """Flat ``{deprecated: canonical}`` view across every target.

        Resolver-layer code (in :mod:`isaaclab_tasks.utils.hydra`) needs
        a target-agnostic lookup -- the ``presets=...`` token is
        target-agnostic on the wire. Builds fresh from per-target tables
        so this enum stays the single source of truth.

        Returns:
            Mapping of every legacy alias to its canonical replacement,
            aggregated across all members.
        """
        return {name: rep for target in cls for name, rep in target.legacy_aliases.items()}


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

        Args:
            target: Which preset target the canonical name belongs to.
                Determines which ``--flag`` accepts it and which lookup
                table the binding lives in.
            name: Canonical name the backend wants to expose. Must be
                unique within *target* across the whole process.

        Returns:
            A class decorator that records the binding and returns the
            class unchanged.

        Raises:
            RuntimeError: ``(target, name)`` is already bound to a
                different class than the one being decorated. Catches
                accidental name reuse across backend packages.
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
            return target_cls

        return deco

    @classmethod
    def names_for(cls, target: PresetTarget) -> set[str]:
        """Canonical names currently bound under *target*.

        Args:
            target: Which target's bindings to read. Backends only land
                in the registry after their cfg module has been
                imported; expect an empty set in a fresh process before
                any task or backend has been touched.

        Returns:
            A fresh set the caller may mutate without disturbing the
            registry.
        """
        return set(cls._entries.get(target, ()))


# Module-level alias kept for the natural decorator spelling at call sites.
register = PresetRegistry.register
"""Decorator alias for :meth:`PresetRegistry.register`."""
