Changed
^^^^^^^

* Unified rough-velocity task inputs across physics backends by removing MJWarp-only actuator armatures,
  using 5,000 G1 training iterations for every backend, and representing shared base-COM randomization as a
  plain event. Downstream configurations that require the former backend-specific behavior should set it
  explicitly.
