* Added the ``mesh_collision_props`` slot to
  :class:`~isaaclab.sim.spawners.RigidObjectSpawnerCfg`, so the USD, URDF, MJCF, shape, and mesh
  spawners take mesh-collision fragments (e.g. :class:`~isaaclab.sim.schemas.UsdPhysicsMeshCollisionCfg`)
  like the mesh-file spawner already did. The slot tunes only colliders: a bare fragment reaches every
  collider under a USD asset and the geometry prim of a shape or mesh spawner, matching the legacy
  ``CollisionBaseCfg.mesh_collision_property`` field it replaces.
