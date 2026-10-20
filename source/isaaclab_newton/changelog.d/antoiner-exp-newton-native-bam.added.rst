* Added :class:`~isaaclab_newton.actuators.BamFriction`, the BAM servo part Newton's native BAM path lacks:
  the gearbox friction budget.
* Added :func:`~isaaclab_newton.actuators.get_bam_group` and :class:`~isaaclab_newton.actuators.BamGroup`
  to read and write a Newton-executed BAM group's friction scale and supply.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.request_solver_observables` to have the solver export
  observables, such as the MuJoCo solver's joint loads, on every substep. The per-DOF observables of reset
  environments read zero until their next solve.
