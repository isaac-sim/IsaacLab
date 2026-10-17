* Fixed cameras attached to moving PhysX bodies keeping their spawn pose in CPU simulation without an Isaac RTX camera.
  :class:`~isaaclab.sensors.camera.Camera` now turns on continuous Fabric transform updates there, as Isaac RTX cameras
  already do, because PhysX on CPU writes body poses to Fabric only if these updates are on when the simulation starts.
