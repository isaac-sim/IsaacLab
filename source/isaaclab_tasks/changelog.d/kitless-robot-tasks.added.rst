* Added G1 and GR1T2 kitless teleoperation setup, including portable GR1T2 kinematics,
  backend-specific actuators, collision preparation, and bounded action targets.
* Added public OvPhysX and Newton selection while retaining Kit defaults and supporting
  standalone backend preparation without optional teleoperation configuration.
  Registered construction prepares the selected backend and preserves prepared configuration copies
  and serialized spawn callbacks.
* Batched G1 joint-limit preparation and device validation without changing action safety bounds.
* Prepared packing-table mesh collisions for standalone physics.
