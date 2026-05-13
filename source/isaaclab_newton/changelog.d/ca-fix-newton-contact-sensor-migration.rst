Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.sensors.ContactSensor` metadata extraction
  after the migration to Newton 1.1, where ``sensing_obj_type`` and
  ``counterpart_type`` became scalar strings and ``counterpart_indices``
  became per-row.
