* Changed :class:`~isaaclab.sim.spawners.DeformableObjectSpawnerCfg` so that setting the deprecated
  ``deformable_props`` field together with ``volume_deformable_props`` or
  ``surface_deformable_props`` no longer raises a ``ValueError``: the legacy field takes precedence
  and the slot is ignored with a warning. This keeps configs that override ``deformable_props`` on
  a preset with a filled slot working. Move the legacy settings into the slot as deformable-body
  fragments and unset ``deformable_props``. Setting both slots still raises a ``ValueError``.
