Changed
^^^^^^^

* **Breaking:** Changed the shipped robot configurations to the composable physics-schema
  fragment API. The ``rigid_props``, ``collision_props``, ``articulation_props`` and
  ``joint_drive_props`` spawner fields now hold a fragment (or a list for multiple fragments) instead of a legacy
  properties config, and the non-USD ``fix_root_link`` and ``ensure_drives_exist`` knobs are set
  on the spawner itself. The authored USD attributes are unchanged, but code that tuned a shipped
  configuration in place must be updated:

  * Single-fragment slots retain direct field access, for example
    ``FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True``.
    When a slot holds multiple fragments, select the one carrying the field, e.g.
    ``next(f for f in cfg.spawn.articulation_props if isinstance(f, PhysxArticulationCfg))`` for
    the PhysX solver iteration counts.
  * Set ``fix_root_link`` and ``ensure_drives_exist`` on the spawner, for example
    ``G1_INSPIRE_FTP_CFG.spawn.fix_root_link = True`` instead of
    ``G1_INSPIRE_FTP_CFG.spawn.articulation_props.fix_root_link = True``.

* Changed articulation self-collision to be authored for both backends, by
  :class:`~isaaclab_physx.sim.schemas.PhysxArticulationCfg` and
  :class:`~isaaclab_newton.sim.schemas.NewtonArticulationCfg`, matching what the legacy writer
  mirrored automatically.
