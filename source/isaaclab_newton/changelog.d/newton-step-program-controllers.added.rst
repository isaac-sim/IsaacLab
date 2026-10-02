* Added :attr:`~isaaclab_newton.physics.StepPhase.COMMAND`, a step-program phase before Newton actuators where
  controllers write joint targets and efforts from the current physics state.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.prepare` and ``runtime.prepare`` to compile and capture the step
  program ahead of the next step, and :meth:`~isaaclab_newton.physics.NewtonManager.require_host_physics_steps` for
  consumers that need host work between physics steps.
* Added a graph-native path to :class:`~isaaclab_newton.envs.mdp.actions.NewtonOperationalSpaceControllerAction`: on
  Newton physics the controller computes efforts inside the captured step program before every physics step, so the
  decimation loop stays folded.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.view_row_worlds` and a ``row_worlds`` argument to
  :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_body_state`, so reset masks map view rows to worlds.
