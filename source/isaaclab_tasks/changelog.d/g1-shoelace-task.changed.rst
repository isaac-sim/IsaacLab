* Centralized shared shoelace grasp settings in reward and success term ``params`` while
  preserving separate G1 and Dual-Franka policy observation layouts. Direct MDP callers must
  pass ``target_diameter``, ``pad_distance_std``, ``aperture_std``, and ``capture_std`` explicitly;
  pregrasp additionally requires ``open_apertures`` and success requires its geometry limits.
* Moved robot, solver, and success settings into the task configuration and fixed cable material
  settings into ``shoelace_physics``; ``shoelace_constants`` retains shared asset and contact conventions.
