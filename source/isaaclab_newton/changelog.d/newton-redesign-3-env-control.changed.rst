* Changed :meth:`NewtonBackend.handles_decimation <isaaclab_newton.physics.NewtonBackend.handles_decimation>` to also
  run the decimation loop in one step when the environment's control is bound.
* Changed Newton articulations, rigid objects and rigid object collections to record their wrench work into a
  captured step only once a wrench composer was used, which captured the step again.
