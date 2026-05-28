Changed
^^^^^^^

* Changed ``teleop_replay_agent.py``'s ``cpu_frame_time_ms`` and ``fps``
  blocks to derive from honest per-rendered-frame samples. The agent
  wraps :meth:`~isaaclab.sim.SimulationContext.render` and records the
  wall-clock interval between successive calls produced from inside
  ``env.step`` during the active window, replacing the prior
  ``env.step``-CPU-time-divided-by-``decimation / render_interval``
  projection. Percentiles now reflect the actual per-frame distribution
  rather than the sum-of-frames smoothing that ``env.step`` totals
  averaged over. The run dict's ``active_iterations`` field now counts
  ``env.step`` calls via a dedicated counter rather than the length of
  the (now per-render) sample list. The agent terminates with a
  ``RuntimeError`` when a run records active ``env.step`` iterations
  but produces no per-render samples (e.g. when the sim is not
  rendering).
