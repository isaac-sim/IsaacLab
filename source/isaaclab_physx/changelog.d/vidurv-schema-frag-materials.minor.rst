Added
^^^^^

* Added :class:`~isaaclab_physx.sim.spawners.materials.PhysxMaterialCfg`, a single-namespace
  ``physxMaterial`` rigid-body physics-material fragment (compliant-contact spring stiffness/damping
  and the friction/restitution combine-mode tokens) backing ``PhysxMaterialAPI``.
* Added :attr:`~isaaclab_physx.sim.spawners.materials.PhysxMaterialCfg.damping_combine_mode` (writes
  ``physxMaterial:dampingCombineMode``) and
  :attr:`~isaaclab_physx.sim.spawners.materials.PhysxMaterialCfg.compliant_contact_acceleration_spring`
  (writes ``physxMaterial:compliantContactAccelerationSpring``), completing the fragment's coverage
  of ``PhysxMaterialAPI``. Also added the same two fields to the legacy
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg`.

Changed
^^^^^^^

* Changed the default physics-material spawn in the PhysX backend to go through
  :func:`~isaaclab.sim.spawners.materials.spawn_physics_material`, so a list of
  :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialFragment` instances assigned to
  :attr:`~isaaclab.sim.SimulationCfg.physics_material` now works when PhysX is the active backend
  (previously only a legacy single cfg was accepted there).
