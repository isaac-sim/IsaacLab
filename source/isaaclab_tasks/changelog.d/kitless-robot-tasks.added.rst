* Added G1 and GR1T2 teleoperation setup for OvPhysX and Newton, including portable GR1T2 kinematics,
  backend-specific actuators, collision preparation, and bounded action targets. OvPhysX and Newton preparation
  replaces the GR1T2 Pink action term with
  :class:`~isaaclab_tasks.contrib.pick_place.gr1t2_bounded_pink_action_cfg.GR1T2BoundedPinkIKActionCfg`;
  Isaac Sim PhysX keeps the Pink term. GR1T2 on Newton uses Newton's collision pipeline with implicit
  integration over two substeps, so the resting steering wheel settles and replays repeat.
* Added public OvPhysX and Newton selection while retaining Kit defaults and supporting
  standalone backend preparation without optional teleoperation configuration.
  Registered construction prepares the selected backend and preserves prepared configuration copies
  and serialized spawn callbacks. Newton preparation keeps the PhysX joint and body order, so
  observations and actions match Kit-recorded data.
* Prepared packing-table mesh collisions for standalone physics.
