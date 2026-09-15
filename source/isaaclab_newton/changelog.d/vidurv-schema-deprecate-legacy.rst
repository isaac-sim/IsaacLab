Deprecated
^^^^^^^^^^

* Deprecated the Newton and MuJoCo schema cfg classes in favor of the single-namespace schema
  fragments. Each class now raises a ``DeprecationWarning`` on instantiation and will be removed in
  4.0. Replace :class:`~isaaclab_newton.sim.schemas.MujocoRigidBodyPropertiesCfg` with
  :class:`~isaaclab_newton.sim.schemas.MujocoRigidBodyCfg`;
  :class:`~isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg` with
  :class:`~isaaclab_newton.sim.schemas.MujocoJointCfg`;
  :class:`~isaaclab_newton.sim.schemas.NewtonCollisionPropertiesCfg` with
  :class:`~isaaclab_newton.sim.schemas.NewtonCollisionCfg`;
  :class:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionPropertiesCfg` with
  :class:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionCfg`;
  :class:`~isaaclab_newton.sim.schemas.NewtonSDFCollisionPropertiesCfg` with
  :class:`~isaaclab_newton.sim.schemas.NewtonSDFCollisionCfg`; and
  :class:`~isaaclab_newton.sim.schemas.NewtonArticulationRootPropertiesCfg` with
  :class:`~isaaclab_newton.sim.schemas.NewtonArticulationCfg`. The empty
  :class:`~isaaclab_newton.sim.schemas.NewtonRigidBodyPropertiesCfg` and
  :class:`~isaaclab_newton.sim.schemas.NewtonJointDrivePropertiesCfg` bases are replaced by the
  corresponding ``UsdPhysics*Cfg`` and ``Mujoco*Cfg`` fragments. Pass fragments as a list in the
  matching spawner slot. The Newton deformable cfgs are unaffected.
