Changed
^^^^^^^

* Changed :meth:`~isaaclab.physics.PhysicsManager.close` to report stored STOP
  listener failures after all listeners run and shared state is cleared. Callers
  that intentionally ignore teardown failures should catch ``RuntimeError``.

Fixed
^^^^^

* Fixed dead weakly referenced listeners to become no-ops instead of raising
  during lifecycle event dispatch.
* Fixed :meth:`~isaaclab.sim.SimulationContext.clear_instance` to finish cleanup
  before reporting STOP-listener failures.
