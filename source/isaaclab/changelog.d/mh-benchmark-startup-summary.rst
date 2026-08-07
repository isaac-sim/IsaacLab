Added
^^^^^

* Added console summaries printed at the end of the startup, runtime, and training benchmark
  workflows: per-phase wall clock (including the timers that ran during environment creation)
  for startup, and throughput, resources, and learning curves for runtime and training. The
  full result remains in the JSON output.

Changed
^^^^^^^

* Changed the benchmark entrypoints to disable Warp adjoint code generation
  (``warp.config.enable_backward``), matching the training and play scripts. This reduces
  cold-start kernel build time.
* Changed the startup benchmark output filename prefix from ``startup_<task>`` to
  ``benchmark_startup_<task>``, matching the runtime, training, and play workflows. Update
  any tooling that globs ``startup_*.json`` to use ``benchmark_startup_*.json``.

Fixed
^^^^^

* Fixed the startup profiling whitelist in ``scripts/benchmarks/startup_whitelist.yaml``,
  whose patterns were written with an ``isaaclab.`` package prefix that profile labels do
  not carry. Every listed pattern silently reported zero time; patterns are now matched
  against the labels the profiler emits.
