* Fixed the ``write_nodal_*_to_sim_mask`` methods of :class:`~isaaclab.assets.BaseDeformableObject`, which the
  OVPhysX deformable object inherits, raising a :class:`RuntimeError` whenever ``env_mask`` was given.
* Fixed :meth:`~isaaclab.sensors.Camera.reset` with an ``env_mask`` on another device than the camera: the camera
  poses are now updated through the mask resolved on the camera's device, like the timestamps and buffers.
