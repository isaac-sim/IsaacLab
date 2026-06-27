Added
^^^^^

* Added the PhysX mesh-collision cooking fragments:
  :class:`~isaaclab_physx.sim.schemas.PhysxConvexHullCfg`,
  :class:`~isaaclab_physx.sim.schemas.PhysxConvexDecompositionCfg`,
  :class:`~isaaclab_physx.sim.schemas.PhysxTriangleMeshCfg`,
  :class:`~isaaclab_physx.sim.schemas.PhysxTriangleMeshSimplificationCfg`, and
  :class:`~isaaclab_physx.sim.schemas.PhysxSDFMeshCfg`. Each is a single-namespace
  :class:`~isaaclab.sim.schemas.MeshCollisionFragment` owning one ``physx*Collision:*`` namespace and
  applied schema, dispatched via :func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`.
