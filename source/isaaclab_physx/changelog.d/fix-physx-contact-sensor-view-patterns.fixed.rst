* Fixed slow :class:`~isaaclab_physx.sensors.ContactSensor` initialization in cloned scenes. Sensors without
  ``filter_prim_paths_expr`` now create their PhysX views from the per-environment body paths instead of one
  pattern per body, which PhysX resolved in every environment.
