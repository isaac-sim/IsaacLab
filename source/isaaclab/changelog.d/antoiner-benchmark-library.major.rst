Added
^^^^^

* Added per-iteration success-rate curves to schema v1.3 training benchmark
  bundles while preserving the existing summary success rate.
* Added shared declarative asset micro-benchmark suites, lazy backend adapters,
  and the ``isaaclab microbenchmark`` component command.

Changed
^^^^^^^

* **Breaking:** Renamed ``--num_frames`` and ``--warmup_frames`` to
  ``--num_steps`` and ``--warmup_steps`` for runtime and play benchmarks, and
  renamed the corresponding ``num_frames`` and ``warmup_frames`` typed request
  fields to ``num_steps`` and ``warmup_steps``. Use the step-based names. Pass
  ``0`` warm-up steps to include every step in timing.

Fixed
^^^^^

* Fixed RSL-RL training benchmarks to report their saved checkpoint and startup
  benchmarks to report their effective environment count.
