Fixed
^^^^^

* Fixed a potential deadlock in :meth:`~isaaclab_physx.assets.SurfaceGripper._initialize_impl`
  by performing the CPU-backend check before loading the upstream
  ``isaacsim.robot.surface_gripper`` extension. Configurations using a non-CPU
  device now fail fast without triggering the extension load and the
  ``SurfaceGripperView`` initialization that can hang in CI.
