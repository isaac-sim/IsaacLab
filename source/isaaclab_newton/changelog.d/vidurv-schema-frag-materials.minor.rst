Added
^^^^^

* Added :class:`~isaaclab_newton.sim.spawners.materials.NewtonMaterialCfg`, a single-namespace
  ``newton`` rigid-body physics-material fragment (torsional and rolling friction) backing
  ``NewtonMaterialAPI``. Composes with other rigid-body material fragments (e.g.
  :class:`~isaaclab.sim.spawners.materials.UsdPhysicsRigidBodyMaterialCfg`) in a fragment list. The
  legacy :class:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg` is unchanged.
