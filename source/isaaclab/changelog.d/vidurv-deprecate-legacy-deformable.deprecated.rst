* Deprecated :class:`~isaaclab.sim.schemas.DeformableBodyPropertiesBaseCfg` in favor of the
  deformable-body schema fragments. It now raises a ``DeprecationWarning`` on instantiation and
  will be removed in 3.2. The class carries no fields: pass
  :class:`~isaaclab.sim.schemas.DeformableBodyFragment` subclasses such as
  :class:`~isaaclab.sim.schemas.OmniPhysicsDeformableBodyCfg` in the spawner's
  ``volume_deformable_props`` or ``surface_deformable_props`` slot, and subclass
  :class:`~isaaclab.sim.schemas.DeformableBodyFragment` for a custom deformable-body cfg.
* Deprecated the ``deformable_props`` field of :class:`~isaaclab.sim.spawners.DeformableObjectSpawnerCfg`
  in favor of ``volume_deformable_props`` and ``surface_deformable_props``; it will be removed in
  3.2. The legacy field authors a surface deformable when ``physics_material`` is a
  :class:`~isaaclab.sim.spawners.materials.SurfaceDeformableBodyMaterialBaseCfg` and a volume
  deformable otherwise, so move its fragments to the matching slot. The field raises no warning of
  its own; the legacy cfgs it takes and the writers it calls do.
* Deprecated :func:`~isaaclab.sim.schemas.define_deformable_body_properties` and
  :func:`~isaaclab.sim.schemas.modify_deformable_body_properties` in favor of
  :func:`~isaaclab.sim.schemas.apply_volume_deformable_properties` and
  :func:`~isaaclab.sim.schemas.apply_surface_deformable_properties`, which take a prim-path
  expression and a list of fragments. Pick the writer that matches the deformable type and pass
  ``create_if_missing=True`` to replace ``define_deformable_body_properties``. Each legacy writer
  now raises a ``DeprecationWarning`` when called and will be removed in 3.2.
  :func:`~isaaclab.sim.schemas.define_deformable_curve_properties` is not deprecated.
