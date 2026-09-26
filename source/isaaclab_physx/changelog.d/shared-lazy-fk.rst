Fixed
^^^^^

* Shared pending kinematic refresh between articulation and scene-data reads so one state write
  did not trigger redundant FK for separate consumers.
