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
* Added :attr:`~isaaclab.sim.spawners.materials.UsdPhysicsRigidBodyMaterialCfg.density` (writes
  ``physics:density``), completing the fragment's coverage of ``UsdPhysics.MaterialAPI``.

Changed
^^^^^^^

* Widened the mesh spawner's rigid-vs-deformable physics-material guard in
  :mod:`isaaclab.sim.spawners.meshes` to accept a
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment` or a list of them (previously
  only the legacy rigid-body material cfg was accepted).
* Widened :attr:`~isaaclab.sim.spawners.from_files.GroundPlaneCfg.physics_material` to accept a
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment` or a list of them, matching
  :attr:`~isaaclab.sim.spawners.from_files.FileCfg.physics_material`.
