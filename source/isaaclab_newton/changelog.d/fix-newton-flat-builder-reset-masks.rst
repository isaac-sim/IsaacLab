Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_fk` and
  :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_body_state` conflating the environment
  index with the world index in their reset-mask kernels. This is correct for the replicated
  builder (one world per environment) but wrote out of bounds for the flat (non-replicated)
  builder used when ``replicate_physics=False``, where every environment shares world 0 — causing
  an "illegal memory access" CUDA error on every ``env.reset()`` call for any non-replicated scene
  with more than one environment.
