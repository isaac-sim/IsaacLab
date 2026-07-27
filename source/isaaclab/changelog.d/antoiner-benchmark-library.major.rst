Added
^^^^^

* Added :mod:`isaaclab.benchmark` as the public benchmark framework and added
  typed Python requests for runtime, startup, training, and play workflows.
* Added an optional serialized synchronized breakdown of time inside and outside
  simulation calls; the outside-simulation remainder is not classified as
  overhead.

Changed
^^^^^^^

* **Breaking:** Replaced the internal :mod:`isaaclab.test.benchmark` API and
  removed backend-specific compatibility scripts. Import :mod:`isaaclab.benchmark`
  and use the supported ``scripts/benchmarks/{runtime,startup,training,play}.py``
  launchers, ``isaaclab benchmark``, or
  :func:`~isaaclab.benchmark.run_benchmark` instead.
* **Breaking:** Standardized benchmark warm-up arguments. Runtime and play use
  ``--warmup_frames``, and :class:`~isaaclab.benchmark.BenchmarkPlayRequest` uses
  ``warmup_frames``. Training continues to use ``--warmup_steps``. Pass the
  corresponding option with a value of ``0`` to include every step in timing.
* Changed runtime benchmarks to report effective throughput over the complete
  measured interval and added environment-step host-return rates across runtime,
  play, and training benchmarks.
