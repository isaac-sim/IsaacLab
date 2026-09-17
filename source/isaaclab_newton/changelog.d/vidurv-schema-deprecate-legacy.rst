Deprecated
^^^^^^^^^^

* Deprecated the Newton and MuJoCo schema cfg classes in favor of the single-namespace schema
  fragments. Each class now raises a ``DeprecationWarning`` on instantiation and will be removed in
  3.1. The warning names *every* fragment the class's fields need, including the fields it inherits
  from a legacy base, so following it does not drop authored properties. Replace
  :class:`~isaaclab_newton.sim.schemas.NewtonRigidBodyPropertiesCfg` with
  ``[UsdPhysicsRigidBodyCfg(...), PhysxRigidBodyCfg(...)]`` and
  :class:`~isaaclab_newton.sim.schemas.MujocoRigidBodyPropertiesCfg` with that pair plus
  :class:`~isaaclab_newton.sim.schemas.MujocoRigidBodyCfg`;
  :class:`~isaaclab_newton.sim.schemas.NewtonJointDrivePropertiesCfg` with
  ``[UsdPhysicsDriveCfg(...), PhysxJointCfg(...)]`` and
  :class:`~isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg` with that pair plus
  :class:`~isaaclab_newton.sim.schemas.MujocoJointCfg`;
  :class:`~isaaclab_newton.sim.schemas.NewtonCollisionPropertiesCfg` with
  ``[UsdPhysicsCollisionCfg(...), PhysxCollisionCfg(...), NewtonCollisionCfg(...)]``;
  :class:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionPropertiesCfg` with those three plus
  :class:`~isaaclab.sim.schemas.UsdPhysicsMeshCollisionCfg` and
  :class:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionCfg`;
  :class:`~isaaclab_newton.sim.schemas.NewtonSDFCollisionPropertiesCfg` with those three plus
  :class:`~isaaclab_newton.sim.schemas.NewtonSDFCollisionCfg`; and
  :class:`~isaaclab_newton.sim.schemas.NewtonArticulationRootPropertiesCfg` with
  ``[PhysxArticulationCfg(...), NewtonArticulationCfg(...)]``. The PhysX fragments appear here
  because ``disable_gravity``, ``max_joint_velocity``, ``contact_offset`` / ``rest_offset`` and
  ``articulation_enabled`` have no other USD home today and Newton's importer reads those
  attributes. Pass fragments as a list in the matching spawner slot. The Newton deformable cfgs are
  unaffected.
