* Added :meth:`~isaaclab_ov.physics.OvPhysxManager.setup_deformable_body` so the
  ``volume_deformable_props`` and ``surface_deformable_props`` spawner slots, and
  :func:`~isaaclab.sim.schemas.apply_volume_deformable_properties` /
  :func:`~isaaclab.sim.schemas.apply_surface_deformable_properties` with ``create_if_missing=True``,
  create deformable bodies on the OvPhysX backend. They previously raised
  ``NotImplementedError`` there. The OmniPhysics anchor schemas are authored exactly as on the Kit
  PhysX backend.
