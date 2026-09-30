* **Breaking:** Removed ``NewtonManager.register_post_step_callback``, ``unregister_post_step_callback``,
  ``register_post_actuator_callback``, and ``register_state_force_callback``. Use
  :meth:`~isaaclab_newton.physics.NewtonManager.add_stage` with :attr:`StepPhase.POST_STEP`,
  :attr:`StepPhase.CONTROL`, or :attr:`StepPhase.SUBSTEP`, and ``remove_stage`` with the returned stage.
* **Breaking:** Removed the solver-manager class hooks (``_build_solver``, ``_create_solver``, ``_step_solver``,
  ``_eval_fk_impl``, ``_reset_solver_internals``, and similar). Implement a
  :class:`~isaaclab_newton.physics.NewtonSolverBinding` and set it as the manager's ``solver_binding`` instead.
* Removed the unused ``NewtonManager.instantiate_builder_from_stage``.
