* Deprecated the PhysX deformable-body cfg classes in favor of the deformable-body schema
  fragments. Each class now raises a ``DeprecationWarning`` on instantiation and will be removed in
  3.2. The warning names every fragment the class's fields need and the spawner slot to put them
  in. Replace :class:`~isaaclab_physx.sim.schemas.OmniPhysicsDeformableBodyPropertiesCfg` with
  :class:`~isaaclab.sim.schemas.OmniPhysicsDeformableBodyCfg`;
  ``PhysXDeformableBodyPropertiesCfg`` with :class:`~isaaclab_physx.sim.schemas.PhysxDeformableBodyCfg`
  plus, for surface deformables, :class:`~isaaclab_physx.sim.schemas.PhysxSurfaceDeformableBodyCfg`;
  and :class:`~isaaclab_physx.sim.schemas.PhysxDeformableBodyPropertiesCfg` and the
  :class:`~isaaclab_physx.sim.schemas.DeformableBodyPropertiesCfg` alias with all three. Put them in
  ``surface_deformable_props`` when the spawner's ``physics_material`` is a surface deformable
  material and in ``volume_deformable_props`` otherwise, which is the type the legacy
  ``deformable_props`` field derived. The legacy cfgs author ``kinematic_enabled=False`` and
  ``solver_position_iteration_count=16`` by default. The fragments leave both unset, which
  simulates the same through the schema fallbacks. Set them explicitly to keep the authored USD
  identical.
