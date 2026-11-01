* Added G1 and GR1T2 kitless teleoperation setup, including portable GR1T2 kinematics,
  backend-specific actuators, collision preparation, and bounded action targets. Kitless preparation
  replaces the GR1T2 Pink action term with
  :class:`~isaaclab_tasks.contrib.pick_place.gr1t2_kitless_action_cfg.GR1T2KitlessPinkInverseKinematicsActionCfg`;
  Isaac Sim PhysX keeps the Pink term.
* Added public OvPhysX and Newton selection while retaining Kit defaults and supporting
  standalone backend preparation without optional teleoperation configuration.
  Registered construction prepares the selected backend and preserves prepared configuration copies
  and serialized spawn callbacks.
* Prepared packing-table mesh collisions for standalone physics.
