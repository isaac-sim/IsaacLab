Added
^^^^^

* Added the rigid-body physics-material "fragment" classes
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment` (marker base) and
  :class:`~isaaclab.sim.spawners.materials.UsdPhysicsRigidBodyMaterialCfg` (solver-common
  ``physics:*`` friction/restitution), plus the family writer
  :func:`~isaaclab.sim.spawners.materials.spawn_rigid_body_material_from_fragments` and the slot
  dispatcher :func:`~isaaclab.sim.spawners.materials.spawn_physics_material`. Spawner
  ``physics_material`` slots now accept a list of single-namespace fragments in addition to the
  legacy material cfg.
