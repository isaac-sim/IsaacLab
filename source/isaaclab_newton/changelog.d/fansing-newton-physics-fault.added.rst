* Added :meth:`~isaaclab_newton.physics.NewtonBackend.get_faulted_worlds`, also on
  :class:`~isaaclab_newton.physics.NewtonManager`, and the
  :func:`~isaaclab_newton.envs.mdp.physics_fault` termination term, which terminates an environment whose body
  state is not finite or, with MuJoCo Warp, whose constraints overflowed the per-world ``njmax`` capacity, so only
  that environment resets while the others keep running. Solvers report their own failures through
  :meth:`~isaaclab_newton.physics.NewtonSolver.flag_faulted_worlds`.
