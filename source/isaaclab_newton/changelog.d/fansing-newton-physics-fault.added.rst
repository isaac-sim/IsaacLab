* Added :meth:`~isaaclab_newton.physics.NewtonBackend.get_non_finite_worlds`, also on
  :class:`~isaaclab_newton.physics.NewtonManager`, and the :func:`~isaaclab_newton.envs.mdp.solver_fault`
  termination term, which terminates an environment whose body pose or velocity is not finite after the last
  step, so only that environment resets while the others keep running.
