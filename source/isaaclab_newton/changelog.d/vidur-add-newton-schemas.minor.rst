Added
^^^^^

* Added :class:`~isaaclab_newton.sim.schemas.MujocoRigidBodyPropertiesCfg` with
  :attr:`gravcomp` for body-level gravity compensation (``mjc:gravcomp``).
* Added :class:`~isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg` with
  :attr:`actuatorgravcomp` for joint-level gravity compensation routing
  (``mjc:actuatorgravcomp`` via ``MjcJointAPI``).
* Added :class:`~isaaclab_newton.sim.schemas.NewtonCollisionPropertiesCfg` with
  :attr:`contact_margin` and :attr:`contact_gap` (``newton:*`` via ``NewtonCollisionAPI``).
* Added :class:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionPropertiesCfg` with
  :attr:`max_hull_vertices` (``newton:maxHullVertices`` via ``NewtonMeshCollisionAPI``).
* Added :class:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg` with
  :attr:`torsional_friction` and :attr:`rolling_friction` (``newton:*`` via ``NewtonMaterialAPI``).
* Added :class:`~isaaclab_newton.sim.schemas.NewtonArticulationRootPropertiesCfg` with
  :attr:`self_collision_enabled` (``newton:selfCollisionEnabled`` via ``NewtonArticulationRootAPI``).
