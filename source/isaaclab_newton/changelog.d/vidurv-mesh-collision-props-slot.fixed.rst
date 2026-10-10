* Fixed the deprecation warnings of :class:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionPropertiesCfg`
  and :class:`~isaaclab_newton.sim.schemas.NewtonSDFCollisionPropertiesCfg` to say which fragments go in
  the spawner's ``collision_props`` slot and which in its ``mesh_collision_props`` slot, and corrected the
  claim that :class:`~isaaclab_newton.sim.schemas.NewtonSDFCollisionCfg` implies the ``sdf``
  approximation token: it authors none, since Newton enables SDF generation from ``NewtonSDFCollisionAPI``.
