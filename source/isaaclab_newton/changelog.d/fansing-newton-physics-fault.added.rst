* Added :meth:`~isaaclab_newton.physics.NewtonManager.get_non_finite_worlds`, the
  :func:`~isaaclab_newton.envs.mdp.solver_fault` termination and the
  :func:`~isaaclab_newton.envs.mdp.zero_reward_on_solver_fault` reward term. Detection covers Newton body,
  joint and particle state and MuJoCo persistent solver buffers, including coupled solver entries.
  Fault queries are evaluated lazily once per published state version and shared by both MDP terms.
  Configure the reward term last to zero faulted rewards before summation and episode accumulation.
  MJWarp recovers individual worlds through the existing reset lifecycle. CPU MuJoCo has global solver
  history, so a fault in its private buffers terminates every world.
