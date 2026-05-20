Added
^^^^^

* Added Newton deformable asset exports under
  :mod:`isaaclab_newton.assets.deformable_object`.
* Added deformable registration hooks to Newton cloning so deformable assets can
  be added per replicated world while their USD proxy meshes are skipped by the
  Newton USD importer.
* Added Newton manager abstraction documentation for adding solver managers and
  custom coupled solvers.

Changed
^^^^^^^

* Changed Newton solver configuration exports so
  :class:`~isaaclab_newton.physics.MJWarpSolverCfg`,
  :class:`~isaaclab_newton.physics.XPBDSolverCfg`,
  :class:`~isaaclab_newton.physics.FeatherstoneSolverCfg`, and
  :class:`~isaaclab_newton.physics.KaminoSolverCfg` are provided from
  :mod:`isaaclab_newton.physics.newton_manager_cfg`.
* Changed :class:`~isaaclab_newton.physics.NewtonCfg` to use
  :class:`~isaaclab_newton.physics.MJWarpSolverCfg` as its explicit default
  solver configuration.
* Changed :class:`~isaaclab_newton.physics.NewtonCfg` validation to reject
  :class:`~isaaclab_newton.physics.MJWarpSolverCfg` configurations that combine
  ``use_mujoco_contacts=True`` with ``collision_cfg``. Remove ``collision_cfg``
  or set ``use_mujoco_contacts=False``.

Fixed
^^^^^

* Fixed Newton Fabric synchronization for deformable particle meshes and
  particle-only scenes.
