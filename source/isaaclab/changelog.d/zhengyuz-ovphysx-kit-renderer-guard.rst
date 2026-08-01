Fixed
^^^^^

* Fixed OvPhysX physics paired with a Kit-based renderer failing inside the OvPhysX
  library with ``Failed to initialize Carbonite and load PhysX plugins``. The kitless
  backend guard only covered the Kit visualizer, so the combination is now rejected up
  front with the supported alternatives, matching the existing OVRTX renderer check.
