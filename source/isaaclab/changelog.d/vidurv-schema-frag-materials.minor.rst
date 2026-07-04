Added
^^^^^

* Added the rigid-body physics-material "fragment" classes
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment` (marker base) and
  :class:`~isaaclab.sim.spawners.materials.UsdPhysicsRigidBodyMaterialCfg` (solver-common
  ``physics:*`` friction/restitution/density), plus the family writer
  :func:`~isaaclab.sim.spawners.materials.spawn_rigid_body_material_from_fragments` and the slot
  dispatcher :func:`~isaaclab.sim.spawners.materials.spawn_physics_material`. Relevant
  ``physics_material`` slots now accept a single fragment or list alongside their legacy cfg form.
  Legacy material cfgs are current-stage-only: the dispatcher raises ``ValueError`` for an explicit
  non-current stage, while fragment-based materials support explicit-stage authoring.

* Added :attr:`~isaaclab.sim.spawners.materials.UsdPhysicsRigidBodyMaterialCfg.density` (writes
  ``physics:density``), completing the fragment's coverage of ``UsdPhysics.MaterialAPI``.
* Added :attr:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg.density`, so the legacy
  rigid material base authors every attribute the fragment authors.

Changed
^^^^^^^

* Narrowed :attr:`~isaaclab.sim.spawners.ShapeCfg.physics_material` from the broad
  :class:`~isaaclab.sim.spawners.materials.PhysicsMaterialCfg` to the rigid material base, a single
  rigid-material fragment, or a list of fragments.
* Extended the already rigid-only :attr:`~isaaclab.sim.spawners.GroundPlaneCfg.physics_material`
  and :attr:`~isaaclab.terrains.TerrainImporterCfg.physics_material` slots from the legacy
  :class:`~isaaclab_physx.sim.spawners.materials.RigidBodyMaterialCfg` type to
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg`, a single
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment`, or a list of fragments.
  Deformable materials remain accepted where deformables can spawn
  (:class:`~isaaclab.sim.spawners.FileCfg` and :class:`~isaaclab.sim.spawners.MeshCfg`).
* Changed the generated-terrain material path to use
  :func:`~isaaclab.sim.spawners.materials.spawn_physics_material`, enabling fragment lists there.

Fixed
^^^^^

* Fixed mesh spawners rejecting valid rigid physics materials: the rigid-material check compared
  against the deprecated leaf class, so canonical
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg` and
  :class:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg` instances raised
  ``ValueError`` on rigid meshes. The check now accepts any
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg`.
* Fixed malformed fragment inputs reaching an opaque legacy ``func`` call. Direct and dispatched
  fragment calls now reject empty, mixed, and non-fragment inputs before creating a material prim.
