Fixed
^^^^^

* Shared pending kinematic refresh between articulation and scene-data reads so one state write
  did not trigger redundant FK for separate consumers.

Changed
^^^^^^^

* Used timestamped buffers for native pose/geometry reads and OVRTX uploads, sharing freshness
  handling between legacy and ovstage transports while preserving retries after failed writes.
