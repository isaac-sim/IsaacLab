Added
^^^^^

* Added :class:`~isaaclab_newton.sim.spawners.materials.NewtonMaterialCfg`, a single-namespace
  ``newton`` rigid-body physics-material fragment (torsional and rolling friction) backing
  ``NewtonMaterialAPI``. Composes with other rigid-body material fragments (e.g.
  :class:`~isaaclab.sim.spawners.materials.UsdPhysicsRigidBodyMaterialCfg`) in a fragment list.
* Added :attr:`~isaaclab_newton.sim.spawners.materials.NewtonMaterialCfg.contact_stiffness`,
  :attr:`~isaaclab_newton.sim.spawners.materials.NewtonMaterialCfg.contact_damping`,
  :attr:`~isaaclab_newton.sim.spawners.materials.NewtonMaterialCfg.contact_friction_gain`, and
  :attr:`~isaaclab_newton.sim.spawners.materials.NewtonMaterialCfg.contact_adhesion` (writing
  ``newton:contactStiffness``, ``newton:contactDamping``, ``newton:contactFrictionGain``, and
  ``newton:contactAdhesion``), matching the per-material contact attributes Newton's USD schema
  resolver reads in place of the deprecated per-shape ``ke``/``kd``/``kf``/``ka`` parameters.
* Added the contact attributes
  (:attr:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg.contact_stiffness`,
  :attr:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg.contact_damping`,
  :attr:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg.contact_friction_gain`, and
  :attr:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg.contact_adhesion`) to
  :class:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg`, matching
  :class:`~isaaclab_newton.sim.spawners.materials.NewtonMaterialCfg`.
