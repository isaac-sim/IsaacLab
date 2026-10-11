* **Breaking:** Ported :class:`~isaaclab_contrib.coupling.CouplerSolverAdapter` and the custom MJWarp and VBD
  coupling adapter to :class:`~isaaclab_newton.physics.NewtonSolver` adapter instances. Each builds the adapters of its
  entries (``entry_adapters``, or ``rigid`` and ``soft``) when it is constructed. The custom
  coupling is now a Newton solver, ``CoupledMJWarpVBDSolver``, that holds both sub-solvers.
* :class:`~isaaclab_contrib.coupling.CouplerSolverAdapter` validates its configuration when constructed and reads its
  entries from that configuration instead of the active physics configuration, and no longer rejects nested adapters by comparing
  ``create_solver`` implementations; an entry whose adapter cannot construct a solver fails at construction.
* **Breaking:** Renamed ``NewtonCouplerManager`` and ``NewtonCoupledMJWarpVBDManager`` to
  ``CouplerSolverAdapter`` and ``CoupledMJWarpVBDSolverAdapter``. Select them through the solver configuration;
  the integration manager remains an independent instance.
