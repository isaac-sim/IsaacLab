Changed
^^^^^^^

* Changed the runtime benchmark to exclude configurable warmup frames and
  report effective throughput over the complete measured interval, and
  report environment-step host-return rates across runtime, play, and training
  benchmarks. Added a ``--warmup_steps`` flag to the play and training
  benchmarks that excludes the first N ``env.step()`` calls from
  environment-step timing and defaults to ``1`` to remove cold start. Runtime
  and play now execute the requested warmup steps before the requested number
  of measured steps; pass ``--warmup_steps 0`` to measure the first step.
  Added an optional serialized
  synchronized breakdown of time inside and outside simulation calls; the
  outside-simulation remainder is not classified as overhead.

Fixed
^^^^^

* Fixed training benchmarks to reject warm-up settings that leave no measured
  steps and close environments
  when training or result processing fails.
* Fixed effective throughput metrics to report the conventional sample standard
  deviation of observed rates.
* Fixed play benchmark environments not closing when policy loading,
  execution, or result processing failed.
