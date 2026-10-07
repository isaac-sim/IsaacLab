* Added support for Newton articulations and rigid objects that exist in only some environments: their gravity and
  reset bookkeeping follow the model worlds of their view, and
  :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_fk` accepts the view's ``world_ids``. Assets whose joint
  or body rows are not regularly spaced between environments raise at initialization, as do Newton-native
  actuators when environments have different joint DOF counts.
