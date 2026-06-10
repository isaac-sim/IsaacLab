Added
^^^^^

* Added :class:`~isaaclab_newton.physics.MPMSolverCfg` and
  :class:`~isaaclab_newton.physics.NewtonMPMManager` for Newton implicit MPM
  simulations.
* Added :class:`~isaaclab_newton.assets.MPMObject` (with
  :class:`~isaaclab_newton.assets.MPMObjectCfg` and
  :class:`~isaaclab_newton.assets.MPMObjectData`) exposing Newton MPM particles
  through the deformable-object interface, together with the declarative
  particle spawner configs :class:`~isaaclab_newton.sim.MPMGridCfg`,
  :class:`~isaaclab_newton.sim.MPMPointsCfg`, and
  :class:`~isaaclab_newton.sim.MPMParticleMaterialCfg`.
* Added the :class:`~isaaclab_newton.physics.NewtonManager` subclass hooks
  ``_register_builder_attributes`` (register a solver's Newton custom builder
  attributes), ``_prepare_builder_for_finalize`` (normalize imported builder
  data before finalization), and ``_supports_cuda_graph_capture`` (opt a solver
  out of CUDA graph capture).
* Added :attr:`~isaaclab_newton.physics.MPMSolverCfg.project_outside_colliders`
  (default ``False``): when set,
  :class:`~isaaclab_newton.physics.NewtonMPMManager` runs
  ``SolverImplicitMPM.project_outside`` after each substep to push particles out
  of collider interiors.
* Added :attr:`~isaaclab_newton.physics.NewtonCfg.simplify_meshes` to control
  whether Newton replication approximates mesh colliders with convex hulls.
  Disable it for thin or hollow MPM colliders that need exact triangle meshes.
* Added ``visual_update_frequency`` to MPM particle spawner configs so Kit USD
  point-cloud visualization can be throttled independently from physics.

Changed
^^^^^^^

* :meth:`~isaaclab_newton.physics.NewtonManager.sync_particles_to_usd` now also
  writes registered ``UsdGeom.Points`` prims (used for MPM particle clouds) in
  addition to the existing Fabric mesh-points sync for deformable visuals.
* :meth:`~isaaclab_newton.physics.NewtonManager.create_builder` and the model
  build path now invoke the active manager's solver-specific builder hooks so
  MPM custom attributes (``mpm:young_modulus``, ...) are registered on the
  builder before particles are added or the model is finalized.
* CUDA graph capture is skipped when the active solver reports it is
  unsupported, so sparse/dense-grid MPM falls back to eager execution.
