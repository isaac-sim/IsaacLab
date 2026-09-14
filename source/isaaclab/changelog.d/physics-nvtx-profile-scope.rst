Added
^^^^^

* Added a synchronized timer around the physics step inside
  :meth:`~isaaclab.sim.simulation_context.SimulationContext.step`, gated by the ``ISAACLAB_PHYSICS_PROFILE``
  environment variable, so any physics backend can be profiled through the same scope name
  (:data:`~isaaclab.sim.simulation_context.PHYSICS_PROFILE_SCOPE`). When enabled, each step prints its elapsed
  time to the log. This mirrors the existing ``ISAACLAB_RENDER_PROFILE`` timer and is off by default, so an
  ordinary run pays neither the print nor the device synchronization.
