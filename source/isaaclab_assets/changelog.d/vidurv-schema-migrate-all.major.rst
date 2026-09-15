Changed
^^^^^^^

* **Breaking:** Changed the shipped robot configurations to the composable physics-schema
  fragment API. The ``rigid_props``, ``collision_props``, ``articulation_props`` and
  ``joint_drive_props`` spawner fields now hold a list of fragments instead of a single legacy
  properties config, and the non-USD ``fix_root_link`` and ``ensure_drives_exist`` knobs are set
  on the spawner itself. The authored USD attributes are unchanged, but code that tuned a shipped
  configuration in place must be updated:

  * Select the fragment that owns the field instead of reaching through the slot, for example
    ``FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props[0].disable_gravity = True`` instead of
    ``FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True``. When a slot holds more
    than one fragment, pick the one carrying the field, e.g.
    ``next(f for f in cfg.spawn.articulation_props if isinstance(f, PhysxArticulationCfg))`` for
    the PhysX solver iteration counts.
  * Set ``fix_root_link`` and ``ensure_drives_exist`` on the spawner, for example
    ``G1_INSPIRE_FTP_CFG.spawn.fix_root_link = True`` instead of
    ``G1_INSPIRE_FTP_CFG.spawn.articulation_props.fix_root_link = True``.

* Changed articulation self-collision to be authored for both backends, by
  :class:`~isaaclab_physx.sim.schemas.PhysxArticulationCfg` and
  :class:`~isaaclab_newton.sim.schemas.NewtonArticulationCfg`, matching what the legacy writer
  mirrored automatically.
