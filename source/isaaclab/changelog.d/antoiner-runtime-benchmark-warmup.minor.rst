Changed
^^^^^^^

* Changed the runtime benchmark to exclude configurable warmup frames and
  report effective throughput over the complete measured interval, and
  report environment-step host-return rates across runtime, play, and training
  benchmarks. Added an opt-in ``--warmup_steps`` flag to the play and training
  benchmarks that excludes the first N ``env.step()`` calls (cold start) from
  the environment-step timing; it defaults to ``0`` (no exclusion). Added an
  optional serialized synchronized breakdown of time inside and outside
  simulation calls; the outside-simulation remainder is not classified as
  overhead.
