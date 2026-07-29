Added
^^^^^

* Added per-iteration success-rate curves to schema v1.3 training benchmark
  bundles while preserving the existing summary success rate.
* Added shared declarative asset micro-benchmark suites, lazy backend adapters,
  and the ``isaaclab microbenchmark`` component command.

Changed
^^^^^^^

* **Breaking:** Renamed ``--warmup_frames`` to ``--warmup_steps`` for runtime
  and play benchmarks, and renamed ``warmup_frames`` to ``warmup_steps`` in
  :class:`~isaaclab.benchmark.BenchmarkRuntimeRequest` and
  :class:`~isaaclab.benchmark.BenchmarkPlayRequest`. Use the step-based names.
  Pass ``0`` to include every step in timing.

Fixed
^^^^^

* Fixed RSL-RL training benchmarks to report their saved checkpoint and startup
  benchmarks to report their effective environment count.
