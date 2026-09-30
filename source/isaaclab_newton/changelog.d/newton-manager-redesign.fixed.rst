* Fixed Newton actuator telemetry and actuator-history resets when worlds hold different DOF layouts. The actuator
  adapter used a uniform per-environment DOF stride, which undersized its computed-effort buffer.
* Fixed consumer stages accumulating duplicates across hard resets; they now share the lifetime of the model they
  bind to.
* Fixed authored-state invalidation of global articulations, which flagged world 0 for a solver reset.
