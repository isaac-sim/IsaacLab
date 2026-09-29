Added
^^^^^

* Rejected :class:`~isaaclab.actuators.BamActuatorCfg` groups with a message naming the required
  native Newton configuration. PhysX's native actuator host adapter did not support BAM.
  MJWarp provided the solver-hosted friction behavior; other native Newton solvers retained the
  controller's torque-level fallback.
  Other supported actuator configurations were unaffected.
