* Centralized shared shoelace grasp settings in reward and success term ``params`` while
  preserving separate G1 and Dual-Franka policy observation layouts. Direct grasp MDP callers must
  pass ``robot_cfgs``, ``target_diameter``, ``pad_distance_std``, ``aperture_std``, and ``capture_std``
  explicitly; success also requires its geometry limits. Pregrasp requires ``robot_cfgs``,
  ``open_apertures``, and ``target_diameter``; its soft-contact and pad-distance parameters were removed.
  Dense reward callers should replace ``approach_robot_cfgs`` with ``robot_cfgs``.
* Aligned Dual-Franka and G1 approach, pregrasp closure gates, and grasp relative speeds with their
  fixed wrist-offset TCPs. Restored pregrasp alignment/closure weights to 0.75/0.25 and removed its
  soft-contact shaping. Grasp quality still checks live finger contacts and surface geometry.
  Existing policies remain loadable; reward values and grasp-based success decisions change.
* Moved robot, solver, and success settings into the task configuration and fixed cable material
  settings into ``shoelace_physics``; ``shoelace_constants`` retains shared asset and contact conventions.
