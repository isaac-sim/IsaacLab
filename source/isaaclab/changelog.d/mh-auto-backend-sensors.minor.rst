Added
^^^^^

* Added the ``global_world_only`` field to :class:`~isaaclab.sensors.RayCasterCfg`, and the
  ``sensor_shape_prim_expr`` and ``filter_shape_prim_expr`` fields to
  :class:`~isaaclab.sensors.ContactSensorCfg`. These configure Newton-backend behavior and are
  ignored by the PhysX and OvPhysX backends, so a single
  :class:`~isaaclab.sensors.RayCasterCfg` or :class:`~isaaclab.sensors.ContactSensorCfg` selects the
  matching backend implementation automatically without a per-task preset.
