Fixed
^^^^^

* Fixed silent backend mismatches when selecting a typed preset that a task
  does not actually declare on its typed config. Names reserved for a typed
  :class:`~isaaclab_tasks.utils.preset_target.PresetTarget` (e.g. the Newton
  physics solvers ``newton_mjwarp`` / ``newton_kamino`` for
  :attr:`~isaaclab_tasks.utils.preset_target.PresetTarget.PHYSICS`) must now
  resolve through that target -- i.e. replace at least one
  :class:`~isaaclab.physics.PhysicsCfg` -- at least once. Selecting such a
  preset on a task where it only matches unrelated scalar or sensor presets
  (e.g. ``presets=newton_mjwarp`` on a task with no Newton physics) now raises
  a descriptive :class:`ValueError` instead of leaving PhysX active while
  applying Newton-tuned values. Reserved names are declared per target in
  :mod:`isaaclab_tasks.utils.preset_target`, so the check generalizes to any
  typed target without hardcoding solver names in the resolver.
