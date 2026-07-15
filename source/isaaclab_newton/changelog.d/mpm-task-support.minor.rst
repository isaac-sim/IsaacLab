Added
^^^^^

* Added scoped Newton builder-world hooks and independent clone-source builder
  copies for tasks that extend replicated Newton worlds.
* Added isolated-world and bounded sparse-grid capacity options to
  :class:`~isaaclab_newton.physics.MPMSolverCfg`.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.reset_solver_state` for
  clearing solver-owned history after selective simulation-state rewrites.

Fixed
^^^^^

* Fixed Newton articulation poses written during environment reset not reaching
  Fabric until a later physics step in declarative MPM scenes.
* Fixed graph-capable Newton solvers being captured before the environment's
  initial reset and added solver preparation and status checks around replay.
* Fixed empty reset masks unnecessarily clearing solver-owned state every
  physics step.
