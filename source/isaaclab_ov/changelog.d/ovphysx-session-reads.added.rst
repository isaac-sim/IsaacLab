* Added :meth:`~isaaclab_ov.physics.OvPhysxManager.invalidate_session_reads`, called before a structural edit made
  through the native ``PhysX`` handle, such as enabling or disabling a rigid body, so the OvPhysX session reads wait for
  replays in flight and capture again.
