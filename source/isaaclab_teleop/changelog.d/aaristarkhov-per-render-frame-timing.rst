Fixed
^^^^^

* Fixed ``teleop_replay_agent.py``'s ``cpu_frame_time_ms`` and ``fps``
  percentiles, which previously projected per-``env.step`` CPU samples
  onto per-render units by dividing by ``decimation / render_interval``.
  Because each ``env.step`` folds multiple physics substeps and rendered
  frames into a single measurement, those sums are CLT-smoothed and
  underreport per-frame hitches the headset wearer / spectator actually
  sees. The agent now wraps
  :meth:`~isaaclab.sim.SimulationContext.render` and records the
  wall-clock interval between successive calls produced from inside
  ``env.step`` during the active window; that per-rendered-frame series
  is the new source for ``cpu_frame_time_ms`` and ``fps``. The run
  dict's ``active_iterations`` field is backed by a dedicated counter.
  Each interval is the wall-clock delta between successive
  ``env.sim.render`` calls, so at least two calls are required per
  run; runs that stepped the env but produced 0 or 1 renders during
  the active window raise ``RuntimeError`` from
  ``_run_single_replay`` (the agent aborts the batch without writing
  a stdout summary or JSON report, so "no JSON output" is an
  unambiguous measurement-failure signal for CI).
