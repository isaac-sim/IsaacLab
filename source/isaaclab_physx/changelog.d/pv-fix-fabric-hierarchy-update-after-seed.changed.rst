* Removed the second Kit update from :meth:`~isaaclab_physx.physics.PhysxManager.play` at simulation start. The
  Fabric frame views now seed consistent transforms, so the first physics step no longer restores authored camera
  poses.
