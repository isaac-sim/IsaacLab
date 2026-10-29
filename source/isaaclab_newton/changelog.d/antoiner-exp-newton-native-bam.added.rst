* Added :class:`~isaaclab_newton.actuators.BamCommandLag` and :class:`~isaaclab_newton.actuators.BamFriction`,
  the BAM servo parts Newton's native BAM path lacks: the per-environment random command lag, written into
  Newton's delay processor, and the gearbox friction budget.
* Added :func:`~isaaclab_newton.actuators.get_bam_group` and :class:`~isaaclab_newton.actuators.BamGroup`
  to read and write a Newton-executed BAM group's friction scale, its batteries' nominal voltage and
  resistance, and to read the supply voltage of its motors.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.request_solver_observables` to have the solver export
  observables, such as the MuJoCo solver's joint loads, on every substep, into
  :attr:`~isaaclab_newton.physics.NewtonBackend.solver_observables`.
* Added the ``observables`` argument of :meth:`~isaaclab_newton.physics.NewtonSolver.step`, which the manager
  passes once a consumer requested observables.
