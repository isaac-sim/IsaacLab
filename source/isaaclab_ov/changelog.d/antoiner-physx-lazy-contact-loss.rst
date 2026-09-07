Fixed
^^^^^

* Fixed :class:`~isaaclab_ov.sensors.ContactSensor` reporting the last in-contact force forever after a body
  left contact on GPU (issue #7613), when the sensor was configured with ``history_length=0`` and its data
  was read less often than every physics step (for example once per policy step with
  ``lazy_sensor_update=True``). PhysX zeroes the net contact force of a body only on the exact physics step
  where its contact is lost, so a lazily refreshed sensor skipped that step. The ovphysx contact bindings
  are now read on every physics step regardless of the history length, while the warp kernels that consume
  the fetched buffers stay lazy.
