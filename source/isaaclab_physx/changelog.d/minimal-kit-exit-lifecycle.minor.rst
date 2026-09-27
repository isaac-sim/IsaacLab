Added
^^^^^

* Added :class:`~isaaclab_physx.app.KitLauncher`, the Isaac Sim / Kit launcher formerly
  ``isaaclab.app.AppLauncher``. :func:`~isaaclab.app.launch_simulation` constructs it; scripts do not.
* Added ``launcher_type`` to :class:`~isaaclab_physx.physics.PhysxCfg`, naming the launcher its runtime needs.
* Added ``set_gravity`` to :class:`~isaaclab_physx.physics.PhysxManager` to set the scene-wide gravity.

Changed
^^^^^^^

* Changed the Isaac RTX renderer to enable ``omni.replicator.core`` itself, so scripts no longer load it.
