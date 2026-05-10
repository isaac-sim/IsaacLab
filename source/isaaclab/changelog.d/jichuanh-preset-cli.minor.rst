Added
^^^^^

* Added :mod:`isaaclab.utils.preset_registry` which exposes
  :class:`~isaaclab.utils.preset_registry.PresetTarget` (closed enum of
  preset categories: ``physics``, ``renderer``, ``domain``) and
  :class:`~isaaclab.utils.preset_registry.PresetRegistry` (a per-target
  ``{name: cls}`` map plus the
  :meth:`~isaaclab.utils.preset_registry.PresetRegistry.register`
  decorator). Backend cfg classes can now declare their canonical preset
  name with ``@register(PresetTarget.PHYSICS, "physx")``, so consumers
  (typed CLI flags, drift lints, ``--help`` listings) can discover the
  available presets without a hard-coded second list. Legacy CLI alias
  normalization (e.g. ``newton`` -> ``newton_mjwarp``) is part of each
  enum member, with deprecation surfaced via :exc:`FutureWarning`.
