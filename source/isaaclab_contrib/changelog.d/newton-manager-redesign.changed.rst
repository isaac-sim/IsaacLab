* **Breaking:** Ported :class:`~isaaclab_contrib.coupling.NewtonCouplerManager` and the custom MJWarp and VBD
  coupling manager to the stateless solver hooks of :class:`~isaaclab_newton.physics.NewtonManager`. The custom
  coupling is now a Newton solver, ``CoupledMJWarpVBDSolver``, that holds both sub-solvers.
