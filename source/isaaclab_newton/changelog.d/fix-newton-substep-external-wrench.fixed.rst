* Fixed external wrenches acting only on the first solver substep when ``num_substeps > 1``, and only on the first
  physics step when Newton runs the decimation loop. :class:`~isaaclab_newton.physics.NewtonManager` now re-applies
  the body forces written before a step on every solver substep of that step.
