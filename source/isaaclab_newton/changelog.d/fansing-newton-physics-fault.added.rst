* Added :meth:`~isaaclab_newton.physics.NewtonManager.get_non_finite_worlds`, the
  :func:`~isaaclab_newton.envs.mdp.solver_fault` termination, the
  :func:`~isaaclab_newton.envs.mdp.zero_reward_on_solver_fault` reward term and the
  :func:`~isaaclab_newton.envs.mdp.zero_final_obs_on_solver_fault` reset event. Detection covers Newton body,
  joint and particle state and MuJoCo persistent solver buffers, including coupled solver entries.
  Fault queries are evaluated lazily once per state timestamp and shared by all three MDP terms.
  Configure the reward term last to zero faulted rewards before summation and episode accumulation, and the
  reset event first to zero the faulted rows of ``extras["final_obs"]``.
  MJWarp recovers individual worlds through the existing reset lifecycle. CPU MuJoCo has global solver
  history, so a fault in its private buffers terminates every world.
