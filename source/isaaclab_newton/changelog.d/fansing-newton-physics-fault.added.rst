* Added :meth:`~isaaclab_newton.physics.NewtonManager.get_non_finite_worlds` and the
  :func:`~isaaclab_newton.envs.mdp.solver_fault` termination. Detection covers Newton body, joint and particle
  state and MuJoCo persistent solver buffers, including coupled solver entries. Configure the termination
  with ``invalid_transition=True`` to mask the corrupted reward and terminal observation.
  MJWarp recovers individual worlds through the existing reset lifecycle. CPU MuJoCo has global solver
  history, so a fault in its private buffers terminates every world.
