* Added instance-owned OvPhysX backend methods for stepping, forward updates, kinematics,
  gravity and warmup, usable without a simulation context. Separated backend configuration
  and scene-data publication from manager orchestration, following the Newton backend design.
  Preserved the manager API and supported independent attached stages.
