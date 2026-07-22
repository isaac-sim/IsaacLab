Changed
^^^^^^^

* Changed the runtime benchmark to exclude configurable warmup frames and
  report effective throughput over the complete measured interval, and
  report environment-step host-return rates across runtime, play, and training
  benchmarks. Added ``--warmup_frames`` to play benchmarks and
  ``--warmup_steps`` to training benchmarks to exclude the first N
  ``env.step()`` calls from environment-step timing. The new options default
  to ``1`` to remove cold start. Runtime and play now execute the requested
  warmup frames before the requested number of measured frames; pass
  ``--warmup_frames 0`` to measure the first frame. Added an optional serialized
  synchronized breakdown of time inside and outside simulation calls; the
  outside-simulation remainder is not classified as overhead.

Fixed
^^^^^

* Fixed training benchmarks to reject warm-up settings that leave no measured
  steps and close environments when training or result processing fails.
* Fixed effective throughput metrics to report the conventional sample standard
  deviation of observed rates.
* Fixed play benchmark environments not closing when policy loading,
  execution, or result processing failed.
