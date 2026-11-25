* Changed :class:`~isaaclab_newton.physics.NewtonBackend` to drop the model changes queued before the solver is
  built, since a new solver already reflects them. They were delivered again at the first step, and a solver that
  ignores a kind of change logged its warning for setup edits.
