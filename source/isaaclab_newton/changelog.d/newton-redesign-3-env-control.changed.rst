* Changed :meth:`NewtonBackend.handles_decimation <isaaclab_newton.physics.NewtonBackend.handles_decimation>` to keep
  the decimation loop in the environment when Newton actuators run in the step but an action term must be applied
  before every physics step and could not be bound.
* Changed Newton articulations, rigid objects and rigid object collections to record their wrench work into a
  captured step only once a wrench composer was used, which captured the step again.
