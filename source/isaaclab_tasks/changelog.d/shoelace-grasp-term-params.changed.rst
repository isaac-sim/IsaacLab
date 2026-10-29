* Centralized shoelace grasp settings in reward and success term ``params`` instead of
  environment attributes and repeated MDP defaults. Direct MDP callers must pass
  ``target_diameter``, ``pad_distance_std``, ``aperture_std``, and ``capture_std`` explicitly;
  pregrasp additionally requires ``open_apertures`` and success requires its geometry limits.
  Kept robot, solver, and success settings in the environment configuration and fixed cable
  material settings beside the physics builder; shared asset and contact conventions remain
  in ``shoelace_constants``. Existing task values and scoring calculations are unchanged.
  Standardized task module constants to uppercase names without a leading underscore.
