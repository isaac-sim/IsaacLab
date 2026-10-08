* Changed :meth:`~isaaclab_newton.physics.NewtonManager.step` to check the solver status before the simulation time
  advances, so a step that fails its check does not publish a new time, and to generate contacts through an
  overridable ``_collide`` hook.
