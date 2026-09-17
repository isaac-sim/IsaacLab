Deprecated
^^^^^^^^^^

* Deprecated the PhysX schema cfg classes in favor of the single-namespace schema fragments. Each
  class now raises a ``DeprecationWarning`` on instantiation and will be removed in 3.1. The warning
  names *every* fragment the class's fields need, including the fields it inherits from a legacy
  base, so following it does not drop authored properties. Replace
  :class:`~isaaclab_physx.sim.schemas.PhysxRigidBodyPropertiesCfg` with
  ``[UsdPhysicsRigidBodyCfg(...), PhysxRigidBodyCfg(...)]``;
  :class:`~isaaclab_physx.sim.schemas.PhysxJointDrivePropertiesCfg` with
  ``[UsdPhysicsDriveCfg(...), PhysxJointCfg(...)]``;
  :class:`~isaaclab_physx.sim.schemas.PhysxCollisionPropertiesCfg` with
  ``[UsdPhysicsCollisionCfg(...), PhysxCollisionCfg(...)]``;
  :class:`~isaaclab_physx.sim.schemas.PhysxArticulationRootPropertiesCfg` with
  :class:`~isaaclab_physx.sim.schemas.PhysxArticulationCfg`; and the
  ``Physx*MeshPropertiesCfg`` cooking classes with their ``Physx*Cfg`` fragments
  (:class:`~isaaclab_physx.sim.schemas.PhysxConvexHullCfg`,
  :class:`~isaaclab_physx.sim.schemas.PhysxConvexDecompositionCfg`,
  :class:`~isaaclab_physx.sim.schemas.PhysxTriangleMeshCfg`,
  :class:`~isaaclab_physx.sim.schemas.PhysxTriangleMeshSimplificationCfg`,
  :class:`~isaaclab_physx.sim.schemas.PhysxSDFMeshCfg`). Pass fragments as a list in the matching
  spawner slot. The PhysX deformable and tendon cfgs are unaffected.
* Reworded the Isaac Lab 2.x schema aliases (``RigidBodyPropertiesCfg``, ``JointDrivePropertiesCfg``,
  ``CollisionPropertiesCfg``, ``ArticulationRootPropertiesCfg``, ``MeshCollisionPropertiesCfg`` and
  the mesh-cooking aliases) to point their ``DeprecationWarning`` at the fragment replacement rather
  than at the now also-deprecated ``Physx*PropertiesCfg`` classes. The material and tendon aliases
  are unaffected.
