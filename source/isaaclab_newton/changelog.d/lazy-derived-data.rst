Changed
^^^^^^^

* Deferred allocation of derived articulation and rigid-object pose, velocity, heading, and
  projected-gravity buffers until first access. Existing timestamp invalidation and
  finite-difference history updates were preserved; no caller changes are required.
