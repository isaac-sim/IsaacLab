* **Breaking:** Ported :class:`~isaaclab_contrib.coupling.CouplerSolverAdapter` and the custom MJWarp and VBD
  coupling adapter to the stateless solver hooks of :class:`~isaaclab_newton.physics.NewtonSolver`. The custom
  coupling is now a Newton solver, ``CoupledMJWarpVBDSolver``, that holds both sub-solvers.
* :class:`~isaaclab_contrib.coupling.CouplerSolverAdapter` reads its entries from the solver configuration passed to
  the builder hooks instead of the active physics configuration, and no longer rejects nested adapters by comparing
  ``create_solver`` implementations; an entry whose adapter cannot construct a solver fails at construction.
* **Breaking:** Renamed ``NewtonCouplerManager`` and ``NewtonCoupledMJWarpVBDManager`` to
  ``CouplerSolverAdapter`` and ``CoupledMJWarpVBDSolverAdapter``. Select them through the solver configuration;
  the integration manager remains an independent instance.
