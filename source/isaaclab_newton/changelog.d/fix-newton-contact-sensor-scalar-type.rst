Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.sensors.ContactSensor` crash when ``sensing_obj_type`` or
  ``counterpart_type`` is a scalar ``Literal["body", "shape"]`` string (Newton >= 1.2) instead
  of a per-row enum array. The sensor name resolution now correctly broadcasts the scalar type
  across all sensing/counterpart indices.
