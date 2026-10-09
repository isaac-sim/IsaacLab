* **Breaking:** Ported :class:`~isaaclab_contrib.coupling.NewtonCouplerManager` and the custom MJWarp and VBD
  coupling manager to the stateless solver hooks of :class:`~isaaclab_newton.physics.NewtonManager`. The custom
  coupling is now a Newton solver, ``CoupledMJWarpVBDSolver``, that holds both sub-solvers.
* :class:`~isaaclab_contrib.coupling.NewtonCouplerManager` reads its entries from the solver configuration passed to
  the builder hooks instead of the active physics configuration, and no longer rejects nested managers by comparing
  ``create_solver`` implementations; an entry whose manager cannot construct a solver fails at construction.
