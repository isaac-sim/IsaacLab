Added
^^^^^

* Added :mod:`isaaclab.utils.preset_registry` exposing
  :class:`~isaaclab.utils.preset_registry.PresetTarget` (closed enum of
  CLI-flag categories: ``physics``, ``renderer``, ``domain``) and
  :class:`~isaaclab.utils.preset_registry.PresetRegistry` with the
  :meth:`~isaaclab.utils.preset_registry.PresetRegistry.register`
  decorator. Backend cfg classes declare themselves once via
  ``@register(PresetTarget.PHYSICS, "physx")``; the typed-flag CLI layer
  in :mod:`isaaclab_tasks.utils.preset_cli` discovers them at import
  time without a hard-coded list. Legacy aliases (e.g. ``newton`` ->
  ``newton_mjwarp``) move into per-target tables on the enum members
  and are aggregated for hydra's resolver via
  :meth:`~isaaclab.utils.preset_registry.PresetTarget.all_legacy_aliases`.
