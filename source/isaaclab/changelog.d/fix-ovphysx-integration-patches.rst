Fixed
^^^^^

* Fixed :class:`~isaaclab.sensors.SensorBase` incorrectly registering a
  PhysX-specific prim deletion callback when using the OvPhysX physics
  backend, which caused an import failure or incorrect event handling at
  runtime.
