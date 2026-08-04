Added
^^^^^

* Added a compact per-phase startup summary printed to the console at the end of the
  ``isaaclab benchmark startup`` workflow, including a breakdown of the timers that ran
  during environment creation. The full profile remains in the JSON output.

Changed
^^^^^^^

* Changed the benchmark entrypoints to disable Warp adjoint code generation
  (``warp.config.enable_backward``), matching the training and play scripts. This reduces
  cold-start kernel build time.

Fixed
^^^^^

* Fixed the startup profiling whitelist in ``scripts/benchmarks/startup_whitelist.yaml``,
  whose patterns were written with an ``isaaclab.`` package prefix that profile labels do
  not carry. Every listed pattern silently reported zero time; patterns are now matched
  against the labels the profiler emits.
