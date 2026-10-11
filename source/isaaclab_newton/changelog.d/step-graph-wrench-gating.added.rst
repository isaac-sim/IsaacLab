* Added :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_step_graph` and
  ``newton_backend.invalidate_step_graph`` to rebuild the step graph when recorded callback work changes.
* Articulations, rigid objects and rigid-object collections record external-wrench work into the step only after a
  wrench composer is first used, which rebuilds the step graph, so assets without external wrenches add no work to
  each physics step.
