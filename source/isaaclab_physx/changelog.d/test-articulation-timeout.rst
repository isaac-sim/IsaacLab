Fixed
^^^^^

* Fixed :class:`~isaaclab_physx.assets.SurfaceGripper` initialization on
  non-CPU simulation backends to raise before loading the surface gripper
  extension, avoiding hangs during startup.
