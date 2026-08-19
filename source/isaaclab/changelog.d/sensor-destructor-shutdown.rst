Fixed
^^^^^

* Stopped :class:`~isaaclab.sensors.SensorBase` and :class:`~isaaclab.sensors.camera.Camera`
  destructors from running cleanup after interpreter shutdown. Previously, an abort that left a
  camera alive could raise ``ImportError: sys.meta_path is None`` from a lazy ``isaaclab.sim``
  import in ``__del__``, which then masked the original exception in logs.
