Added
^^^^^

* Added forwarding shims on :mod:`isaaclab.sim.schemas` and :mod:`isaaclab.sim` for the
  Newton/MuJoCo cfg classes added in :mod:`isaaclab_newton.sim.schemas`
  (:class:`NewtonRigidBodyPropertiesCfg`, :class:`NewtonJointDrivePropertiesCfg`,
  :class:`NewtonCollisionPropertiesCfg`, :class:`NewtonMeshCollisionPropertiesCfg`,
  :class:`NewtonMaterialPropertiesCfg`, :class:`NewtonArticulationRootPropertiesCfg`,
  :class:`MujocoRigidBodyPropertiesCfg`, :class:`MujocoJointDrivePropertiesCfg`).
  The shims resolve lazily on first access so importing :mod:`isaaclab.sim.schemas`
  does not require :mod:`isaaclab_newton` to be installed.
