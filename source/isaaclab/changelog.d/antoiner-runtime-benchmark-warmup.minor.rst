Changed
^^^^^^^

* Changed the runtime benchmark to exclude configurable warmup frames and
  report effective throughput over the complete measured interval, and
  report environment-step host-return rates across runtime, play, and training
  benchmarks. Added an optional serialized synchronized breakdown of time
  inside and outside simulation calls; the outside-simulation remainder is not
  classified as overhead.
