Fixed
^^^^^

* Fixed coupled Implicit MPM environment resets raising
  ``ValueError: world_mask has shape ...`` or
  ``RuntimeError: Masked reset cannot selectively clear grid-backed warm
  starts`` when :meth:`~isaaclab_newton.physics.NewtonManager.reset_solver_state`
  forwarded Isaac Lab's ``(world_count,)`` mask. MPM entry ``reset`` now receives
  the ``(world_count + 1,)`` mask required by
  :meth:`newton.solvers.SolverImplicitMPM.reset` (or skips selective shared-world
  resets), while MJWarp entries keep the original parent mask.
