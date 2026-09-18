Added
^^^^^

* Added the PhysX deformable-body fragments :class:`~isaaclab_physx.sim.schemas.PhysxDeformableBodyCfg`
  and :class:`~isaaclab_physx.sim.schemas.PhysxSurfaceDeformableBodyCfg`, covering the solver,
  damping, and self-collision attributes from ``PhysxBaseDeformableBodyAPI`` and the
  surface-only collision attributes from ``PhysxSurfaceDeformableBodyAPI``.
* Added the PhysX deformable material fragments
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxDeformableMaterialCfg` and
  :class:`~isaaclab_physx.sim.spawners.materials.PhysxSurfaceDeformableMaterialCfg`, authoring
  ``physxDeformableMaterial:*`` attributes from ``PhysxDeformableMaterialAPI`` and
  ``PhysxSurfaceDeformableMaterialAPI``.
* Added :meth:`~isaaclab_physx.physics.PhysxManager.setup_deformable_body`, applying the
  OmniPhysics deformable sim and body anchor APIs, rest state, and visual bind pose to a prepared
  deformable mesh.
