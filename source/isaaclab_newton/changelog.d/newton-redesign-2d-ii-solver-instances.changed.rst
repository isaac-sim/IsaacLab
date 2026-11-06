* **Breaking:** Changed :class:`~isaaclab_newton.physics.NewtonSolver` from a stateless class of classmethods into a
  driver built from its configuration and model, so that callers no longer pass the configuration and Newton's solver
  to every call. :meth:`~isaaclab_newton.physics.NewtonBackend.init_solver` stored it as ``NewtonBackend.solver_ops``,
  while ``NewtonBackend.solver`` stayed Newton's solver. Builder-time hooks such as ``prepare_builder`` stayed
  classmethods. Custom solvers override ``build(self, model, mode, env_template)`` and instance methods such as
  ``pre_step(self, state)``, reading ``self.cfg`` and ``self.native``.
