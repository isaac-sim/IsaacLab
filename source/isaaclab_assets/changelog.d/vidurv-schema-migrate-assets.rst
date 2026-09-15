Changed
^^^^^^^

* Migrated the shipped robot configurations to the composable physics-schema fragment API.
  The ``rigid_props``, ``collision_props``, ``articulation_props`` and ``joint_drive_props``
  spawner fields now hold a list of fragments instead of a single legacy properties cfg, and
  the non-USD ``fix_root_link`` and ``ensure_drives_exist`` knobs are set on the spawner
  itself. The authored USD attributes are unchanged. Code that tuned a shipped configuration
  in place must index the fragment, for example
  ``FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props[0].disable_gravity = True`` instead of
  ``FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True``, and
  ``G1_INSPIRE_FTP_CFG.spawn.fix_root_link = True`` instead of
  ``G1_INSPIRE_FTP_CFG.spawn.articulation_props.fix_root_link = True``.
