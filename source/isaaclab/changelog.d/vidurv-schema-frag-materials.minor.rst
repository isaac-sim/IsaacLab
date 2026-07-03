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
* Added :attr:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg.density`, so the legacy
  rigid material base authors every attribute the fragment authors.

Changed
^^^^^^^

* Changed rigid-only ``physics_material`` slots
  (:class:`~isaaclab.sim.spawners.ShapeCfg`, :class:`~isaaclab.sim.spawners.GroundPlaneCfg`,
  :class:`~isaaclab.terrains.TerrainImporterCfg`, :class:`~isaaclab.sim.SimulationCfg`) to accept
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` or a list of
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment` instances, and no longer
  advertise deformable material configs on rigid-only spawners. Deformable materials remain
  accepted where deformables can spawn (:class:`~isaaclab.sim.spawners.FileCfg`,
  :class:`~isaaclab.sim.spawners.MeshCfg`).
* Changed the generated-terrain, simulation default-material, and compliant-contact material paths
  to spawn through :func:`~isaaclab.sim.spawners.materials.spawn_physics_material`, so fragment
  lists work at every ``physics_material`` entry point.
* Changed :func:`~isaaclab.sim.spawners.materials.spawn_physics_material` to raise ``ValueError``
  when a legacy material cfg is given an explicit stage other than the current stage (previously
  the stage was silently ignored and authored on the wrong stage). Callers that already pass the
  current stage (or ``None``) are unaffected; callers that need an explicit non-current stage
  should switch the material to the fragment-based API, which supports it.

Fixed
^^^^^

* Fixed mesh spawners rejecting valid rigid physics materials: the rigid-material check compared
  against the deprecated leaf class, so canonical
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg` and
  :class:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg` instances raised
  ``ValueError`` on rigid meshes. The check now accepts any
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg`.
