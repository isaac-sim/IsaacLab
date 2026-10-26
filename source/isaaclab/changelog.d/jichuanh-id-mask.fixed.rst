* Fixed the ``write_nodal_*_to_sim_mask`` methods of :class:`~isaaclab.assets.BaseDeformableObject`, which the
  OVPhysX deformable object inherits, raising a :class:`RuntimeError` whenever ``env_mask`` was given.
