Fixed
^^^^^

* Shared pending kinematic refresh between articulation and scene-data reads so one state write
  did not trigger redundant FK for separate consumers.

Changed
^^^^^^^

* Used timestamped buffers for native pose/geometry reads and Fabric geometry destinations,
  retaining native geometry batches across reads and preserving rendering cadence.
