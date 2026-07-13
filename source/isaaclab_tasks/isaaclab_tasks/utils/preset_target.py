# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed enum of typed preset categories with per-target metadata.

Each :class:`PresetTarget` member carries everything the preset CLI layer
needs to know about that category in one place:

* ``label`` -- the Hydra-style selector key (e.g. ``"physics"`` for
  ``physics=NAME``) and ``self.value``.
* ``base_classes`` -- the cfg base classes whose subclass instances belong to
  this bucket. Help-time bucketing in :mod:`isaaclab_tasks.utils.preset_cli`
  routes variants by ``isinstance`` against these. Empty for
  :attr:`PresetTarget.DOMAIN`, which is the catch-all whose membership is
  "no typed target matched".
* ``legacy_aliases`` -- deprecated-name to canonical-name table for this
  target, aggregated for hydra's resolver via :meth:`all_legacy_aliases`.

Adding a new typed target = appending one enum member with its label, base
classes, and (optional) legacy alias map. The CLI layer needs no other wiring.
"""

from __future__ import annotations

import enum
import functools
import importlib


class PresetTarget(enum.Enum):
    """Typed preset categories.

    **Bucketing contract.** Help-time bucketing in
    :mod:`isaaclab_tasks.utils.preset_cli` routes each preset variant to a
    typed target by checking ``isinstance(cfg_value, target.base_classes)``
    against every typed target's bases. A variant whose cfg value does *not*
    subclass any typed target's base falls into :attr:`DOMAIN` and shows up
    under the ``presets:`` catch-all in ``--help``.

    To opt into the typed ``physics`` / ``renderer`` help-text listing,
    a backend's cfg class must subclass :class:`~isaaclab.physics.PhysicsCfg`
    or :class:`~isaaclab.renderers.renderer_cfg.RendererCfg` respectively.
    A variant whose class does *not* subclass either base still **resolves
    correctly at runtime** -- hydra applies the selected name across every
    matching ``PresetCfg`` field regardless of class; the typed bucketing only
    governs which header it appears under in ``--help``.

    Adding a new target = appending one enum member.
    """

    # Members. Tuple values are (label, base_class_paths, legacy_aliases); the
    # enum metaclass collects the whole namespace before constructing members,
    # so ``__new__`` below unpacks each tuple regardless of declaration order.
    PHYSICS = (
        "physics",
        ("isaaclab.physics:PhysicsCfg",),
        {"newton": "newton_mjwarp", "kamino": "newton_kamino"},
    )
    """Physics backends -- ``physics=NAME`` selector.

    Legacy aliases ``newton`` -> ``newton_mjwarp`` and ``kamino`` -> ``newton_kamino``
    exist because Newton-backend solver presets were renamed to use the
    ``newton_`` prefix so they group together in autocomplete and read
    distinctly from backend / package / visualizer names that also contain the
    word ``newton``. Hydra's resolver (see
    :func:`~isaaclab_tasks.utils.hydra._normalize_preset_name`) consults these
    and emits a :class:`FutureWarning`; the aliases will be removed in a
    future release.
    """

    RENDERER = (
        "renderer",
        ("isaaclab.renderers.renderer_cfg:RendererCfg",),
        {"ovrtx_renderer": "ovrtx", "isaacsim_rtx_renderer": "isaacsim_rtx"},
    )
    """Camera-sensor renderers -- ``renderer=NAME`` selector.

    Legacy aliases ``ovrtx_renderer`` -> ``ovrtx`` and
    ``isaacsim_rtx_renderer`` -> ``isaacsim_rtx`` exist because the renderer
    presets dropped their redundant ``_renderer`` suffix (the ``renderer=``
    selector already names the category). Hydra's resolver (see
    :func:`~isaaclab_tasks.utils.hydra._normalize_preset_name`) consults these
    and emits a :class:`FutureWarning`; the aliases will be removed in a
    future release.
    """

    DOMAIN = ("presets",)
    """Free-form env-specific presets -- ``presets=NAME[,...]`` selector (catch-all).

    No ``base_classes`` -- any variant whose cfg class doesn't subclass a typed
    target's base ends up here. The ``presets=`` token also acts as a
    broadcast: hydra's resolver applies a DOMAIN-bucketed name to every
    matching ``PresetCfg`` regardless of target. ``self.value`` matches the
    CLI selector key (``"presets"``) so the CLI layer can dispatch by
    enum value without a hardcoded constant.
    """

    def __new__(
        cls,
        label: str,
        base_class_paths: tuple[str, ...] = (),
        legacy_aliases: dict[str, str] | None = None,
    ):
        """Construct a member from its ``(label, base_class_paths, legacy_aliases)`` tuple.

        Args:
            label: Hydra-style selector key (e.g. ``"physics"`` is recognized
                as the ``physics=NAME`` token and becomes ``self.value``).
            base_class_paths: Import paths for cfg base classes whose instances
                route to this target via :func:`isinstance`. Defaults to ``()``
                (no typed routing).
            legacy_aliases: Optional deprecated-to-canonical map for this
                target; copied so members cannot alias each other's tables.

        Returns:
            A new enum member with ``_value_`` set to *label*, plus
            ``base_classes`` and ``legacy_aliases`` attributes.
        """
        obj = object.__new__(cls)
        obj._value_ = label
        obj._base_class_paths = tuple(base_class_paths)
        obj._base_classes = None
        obj.base_class_names = tuple(path.rsplit(":", 1)[-1] for path in base_class_paths)
        obj.is_typed = bool(base_class_paths)
        obj.legacy_aliases = dict(legacy_aliases) if legacy_aliases else {}
        return obj

    @property
    def base_classes(self) -> tuple[type, ...]:
        """Cfg base classes whose instances route to this target.

        The classes are imported lazily so normal CLI parsing can use selector
        metadata without importing backend config modules, torch, or Warp.
        """
        if self._base_classes is None:
            self._base_classes = tuple(_import_symbol(path) for path in self._base_class_paths)
        return self._base_classes

    @classmethod
    @functools.cache
    def all_legacy_aliases(cls) -> dict[str, str]:
        """Flat ``{deprecated: canonical}`` view across every target.

        Resolver-layer code (in :mod:`isaaclab_tasks.utils.hydra`) needs a
        target-agnostic lookup -- the ``presets=...`` token is target-agnostic
        on the wire. Cached because per-member tables are immutable after
        class construction, so the merged view never changes; this keeps
        each lookup O(1) instead of rebuilding on every membership test or
        ``[]`` access. Callers must not mutate the returned dict.

        Returns:
            Mapping of every legacy alias to its canonical replacement,
            aggregated across all members.
        """
        return {name: rep for target in cls for name, rep in target.legacy_aliases.items()}


def _import_symbol(path: str) -> type:
    """Import ``module:attribute`` and return the attribute."""
    module_name, attr_name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), attr_name)
