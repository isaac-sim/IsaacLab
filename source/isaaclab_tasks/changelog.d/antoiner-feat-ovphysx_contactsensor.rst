Added
^^^^^

* Added ``ovphysx`` preset to ``isaaclab_tasks.manager_based.locomotion.velocity``
  for use under the OVPhysX backend. ``AnymalDFlatPhysicsCfg`` now exposes
  an ``ovphysx`` member, and the shared ``LocomotionVelocityRoughEnvCfg``
  injects the OVPhysX :class:`~isaaclab_ovphysx.sensors.ContactSensorCfg`
  alongside the existing PhysX and Newton entries so the velocity task
  selects the right contact sensor backend when run with
  ``presets=ovphysx``.
