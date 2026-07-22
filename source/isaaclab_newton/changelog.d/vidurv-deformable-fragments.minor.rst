Added
^^^^^

* Added :class:`~isaaclab_newton.sim.schemas.NewtonDeformableBodyCfg`, a placeholder deformable-body
  fragment reserving the ``newton:*`` namespace until Newton registers deformable-body attributes.
* Added the Newton deformable material fragments
  :class:`~isaaclab_newton.sim.spawners.materials.NewtonVolumeDeformableMaterialCfg` and
  :class:`~isaaclab_newton.sim.spawners.materials.NewtonSurfaceDeformableMaterialCfg`, authoring the
  ``newton:*`` attributes read by the Newton deformable-body builder hooks.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.setup_deformable_body`, applying Newton's
  token deformable anchor schemas and syncing the visual mesh geometry from the simulation mesh.
